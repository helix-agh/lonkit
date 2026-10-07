import warnings

import numpy as np
import pandas as pd
import pytest

from lonkit import (
    LON,
    BasinHoppingSampler,
    ILSSampler,
    ILSSamplerConfig,
    LONConfig,
    NKLandscape,
    validate_trace,
)
from tests.conftest import DEFAULT_CONFIG, DOMAIN_2D, SEED, rastrigin


def make_trace() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [1, 1, 2],
            "fit1": [12.0, 9.0, 15.0],
            "node1": ["0110", "0111", "1000"],
            "fit2": [9.0, 4.0, 9.0],
            "node2": ["0111", "1111", "0111"],
        }
    )


class TestColumns:
    def test_valid_trace_passes_unchanged(self) -> None:
        trace = make_trace()
        pd.testing.assert_frame_equal(validate_trace(trace), trace)

    def test_columns_matched_by_name(self) -> None:
        trace = make_trace()
        shuffled = trace[["node2", "run", "fit2", "node1", "fit1"]].assign(extra=0)
        pd.testing.assert_frame_equal(validate_trace(shuffled), trace)

    def test_unnamed_columns_taken_positionally(self) -> None:
        trace = make_trace()
        unnamed = trace.set_axis(["a", "b", "c", "d", "e"], axis=1)
        pd.testing.assert_frame_equal(validate_trace(unnamed), trace)

    def test_wrong_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="must contain columns"):
            validate_trace(make_trace().drop(columns="fit2"))

    def test_empty_trace_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_trace(make_trace().iloc[0:0])


class TestValues:
    def test_missing_run_raises(self) -> None:
        trace = make_trace()
        trace["run"] = trace["run"].astype(float)
        trace.loc[1, "run"] = np.nan
        with pytest.raises(ValueError, match=r"Missing run values in rows \[1\]"):
            validate_trace(trace)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, "abc"])
    def test_invalid_fitness_raises(self, bad: object) -> None:
        trace = make_trace().astype({"fit1": object})
        trace.loc[2, "fit1"] = bad
        with pytest.raises(ValueError, match=r"column 'fit1' in rows \[2\]"):
            validate_trace(trace)

    def test_numeric_strings_fitness_converted(self) -> None:
        trace = make_trace().astype({"fit2": str})
        assert validate_trace(trace)["fit2"].dtype == float

    @pytest.mark.parametrize("bad", [None, "", "  "])
    def test_invalid_node_raises(self, bad: object) -> None:
        trace = make_trace()
        trace.loc[0, "node2"] = bad
        with pytest.raises(ValueError, match=r"column 'node2' in rows \[0\]"):
            validate_trace(trace)

    def test_node_ids_converted_to_str(self) -> None:
        trace = make_trace()
        trace["node1"] = [110, 111, 1000]
        trace["node2"] = [111, 1111, 111]
        validated = validate_trace(trace)
        assert validated["node1"].tolist() == ["110", "111", "1000"]
        assert validated["node2"].tolist() == ["111", "1111", "111"]


class TestTrajectoryContinuity:
    def test_shuffled_rows_warn(self) -> None:
        trace = make_trace().iloc[[1, 0, 2]]
        with pytest.warns(UserWarning, match="1 broken trajectory step"):
            validate_trace(trace)

    def test_shuffled_rows_raise_in_strict_mode(self) -> None:
        trace = make_trace().iloc[[1, 0, 2]]
        with pytest.raises(ValueError, match="broken trajectory"):
            validate_trace(trace, strict=True)

    def test_interleaved_runs_are_continuous(self) -> None:
        trace = make_trace().iloc[[0, 2, 1]]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_trace(trace)

    def test_self_loops_are_continuous(self) -> None:
        trace = pd.concat([make_trace().iloc[[0]], make_trace().iloc[[0]].assign(node1="0111")])
        trace = trace.assign(node2=["0111", "0111"], fit1=[12.0, 9.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_trace(trace)


class TestFromTraceData:
    def test_validation_errors_propagate(self) -> None:
        trace = make_trace()
        trace.loc[0, "fit1"] = np.nan
        with pytest.raises(ValueError, match="column 'fit1'"):
            LON.from_trace_data(trace)

    def test_strict_mode(self) -> None:
        trace = make_trace().iloc[[1, 0, 2]]
        with pytest.raises(ValueError, match="broken trajectory"):
            LON.from_trace_data(trace, config=LONConfig(trace_validation="strict"))

    def test_validation_off_keeps_positional_columns(self) -> None:
        trace = make_trace().iloc[[1, 0, 2]]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lon = LON.from_trace_data(trace, config=LONConfig(trace_validation="off"))
        assert lon.n_vertices == 4

    def test_final_run_values_use_validated_columns(self) -> None:
        trace = make_trace()[["node2", "run", "fit2", "node1", "fit1"]]
        lon = LON.from_trace_data(trace)
        assert lon.final_run_values.to_dict() == {1: 4.0, 2: 9.0}
        assert lon.best_fitness == 4.0

    def test_csv_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "trace.csv"
        make_trace().to_csv(path, index=False)
        trace = pd.read_csv(path, dtype={"node1": str, "node2": str})
        lon = LON.from_trace_data(trace)
        assert sorted(lon.vertex_names) == ["0110", "0111", "1000", "1111"]


class TestBuiltinSamplers:
    def test_basin_hopping_trace_passes_strict_validation(self) -> None:
        result = BasinHoppingSampler(DEFAULT_CONFIG).sample(rastrigin, DOMAIN_2D)
        validate_trace(result.trace_df, strict=True)

    @pytest.mark.parametrize("accept_equal", [True, False])
    def test_ils_trace_passes_strict_validation(self, accept_equal: bool) -> None:
        sampler = ILSSampler(
            ILSSamplerConfig(n_runs=5, max_iter=50, accept_equal=accept_equal, seed=SEED)
        )
        result = sampler.sample(NKLandscape(n=20, k=4, instance_seed=SEED))
        assert not result.trace_df.empty
        validate_trace(result.trace_df, strict=True)
