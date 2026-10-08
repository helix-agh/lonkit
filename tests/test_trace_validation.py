import warnings

import numpy as np
import pandas as pd
import pytest

from lonkit import LON, BasinHoppingSampler, ILSSampler, ILSSamplerConfig, NKLandscape
from lonkit.lon import _validate_trace
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


def validate_without_warnings(trace: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _validate_trace(trace)


class TestColumns:
    def test_valid_trace_passes_unchanged(self) -> None:
        trace = make_trace()
        pd.testing.assert_frame_equal(validate_without_warnings(trace), trace)

    def test_columns_in_any_order(self) -> None:
        trace = make_trace()
        shuffled = trace[["node2", "run", "fit2", "node1", "fit1"]]
        pd.testing.assert_frame_equal(_validate_trace(shuffled), trace)

    def test_unnamed_columns_raise(self) -> None:
        unnamed = make_trace().set_axis(["a", "b", "c", "d", "e"], axis=1)
        with pytest.raises(ValueError, match="must have exactly the columns"):
            _validate_trace(unnamed)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="must have exactly the columns"):
            _validate_trace(make_trace().drop(columns="fit2"))

    def test_extra_column_raises(self) -> None:
        with pytest.raises(ValueError, match="must have exactly the columns"):
            _validate_trace(make_trace().assign(iteration=0))

    def test_duplicated_column_raises(self) -> None:
        trace = pd.concat([make_trace(), make_trace()[["run"]]], axis=1)
        with pytest.raises(ValueError, match="must have exactly the columns"):
            _validate_trace(trace)

    def test_empty_trace_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_trace(make_trace().iloc[0:0])


class TestMissingValues:
    @pytest.mark.parametrize("col", ["run", "fit1", "node2"])
    def test_missing_value_raises(self, col: str) -> None:
        trace = make_trace().astype({col: object})
        trace.loc[1, col] = None
        with pytest.raises(ValueError, match=rf"Missing values in columns \['{col}'\]"):
            _validate_trace(trace)


class TestFitness:
    def test_infinite_fitness_raises(self) -> None:
        trace = make_trace()
        trace.loc[2, "fit1"] = -np.inf
        with pytest.raises(ValueError, match="Infinite fitness values in column 'fit1'"):
            _validate_trace(trace)

    def test_nat_fitness_raises(self) -> None:
        # Regression: pd.to_numeric() turned NaT into -9.2e18, which passed validation
        trace = make_trace()
        trace["fit1"] = pd.to_datetime(pd.Series([None, None, None]))
        with pytest.raises(ValueError, match=r"Missing values in columns \['fit1'\]"):
            _validate_trace(trace)

    @pytest.mark.parametrize(
        "values",
        [
            pd.to_datetime(pd.Series(["2020-01-01"] * 3)),
            pd.to_timedelta(pd.Series([1, 2, 3]), unit="s"),
            pd.Series([True, False, True]),
            pd.Series(["12.0", "9.0", "15.0"]),
        ],
    )
    def test_non_numeric_fitness_raises(self, values: pd.Series) -> None:
        trace = make_trace()
        trace["fit2"] = values
        with pytest.raises(ValueError, match="'fit2' must contain numeric fitness values"):
            _validate_trace(trace)

    def test_integer_fitness_converted_to_float(self) -> None:
        trace = make_trace().astype({"fit1": int})
        assert _validate_trace(trace)["fit1"].dtype == float


class TestNodeIds:
    def test_integer_ids_converted_to_str(self) -> None:
        trace = make_trace()
        trace["node1"] = [110, 111, 1000]
        trace["node2"] = [111, 1111, 111]
        validated = validate_without_warnings(trace)
        assert validated["node1"].tolist() == ["110", "111", "1000"]
        assert validated["node2"].tolist() == ["111", "1111", "111"]

    def test_mixed_id_types_raise(self) -> None:
        # Mixing would make e.g. 123 and "123" the same node
        trace = make_trace()
        trace["node1"] = pd.Series([123, "456", 7], dtype=object)
        with pytest.raises(ValueError, match="all strings or all integers"):
            _validate_trace(trace)

    @pytest.mark.parametrize("values", [pd.Series([1.0, 2.0, 3.0]), pd.Series([True, False, True])])
    def test_unsupported_id_types_raise(self, values: pd.Series) -> None:
        trace = make_trace()
        trace["node1"] = values
        trace["node2"] = values
        with pytest.raises(ValueError, match="all strings or all integers"):
            _validate_trace(trace)


class TestTrajectoryContinuity:
    def test_shuffled_rows_warn(self) -> None:
        trace = make_trace().iloc[[1, 0, 2]]
        with pytest.warns(UserWarning, match="1 broken trajectory step"):
            _validate_trace(trace)

    def test_interleaved_runs_are_continuous(self) -> None:
        validate_without_warnings(make_trace().iloc[[0, 2, 1]])

    def test_self_loops_are_continuous(self) -> None:
        trace = pd.concat([make_trace().iloc[[0]], make_trace().iloc[[0]].assign(node1="0111")])
        trace = trace.assign(node2=["0111", "0111"], fit1=[12.0, 9.0])
        validate_without_warnings(trace)


class TestFromTraceData:
    def test_validation_errors_propagate(self) -> None:
        trace = make_trace()
        trace.loc[0, "fit1"] = np.nan
        with pytest.raises(ValueError, match=r"Missing values in columns \['fit1'\]"):
            LON.from_trace_data(trace)

    def test_broken_trajectory_warns(self) -> None:
        with pytest.warns(UserWarning, match="broken trajectory"):
            LON.from_trace_data(make_trace().iloc[[1, 0, 2]])

    def test_final_run_values_use_named_columns(self) -> None:
        trace = make_trace()[["node2", "run", "fit2", "node1", "fit1"]]
        lon = LON.from_trace_data(trace)
        assert lon.final_run_values is not None
        assert lon.final_run_values.to_dict() == {1: 4.0, 2: 9.0}
        assert lon.best_fitness == 4.0

    def test_csv_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "trace.csv"
        make_trace().to_csv(path, index=False)
        trace = pd.read_csv(path, dtype={"node1": str, "node2": str})
        lon = LON.from_trace_data(trace)
        assert sorted(lon.vertex_names) == ["0110", "0111", "1000", "1111"]


class TestBuiltinSamplers:
    def test_basin_hopping_trace_is_valid(self) -> None:
        result = BasinHoppingSampler(DEFAULT_CONFIG).sample(rastrigin, DOMAIN_2D)
        validate_without_warnings(result.trace_df)

    @pytest.mark.parametrize("accept_equal", [True, False])
    def test_ils_trace_is_valid(self, accept_equal: bool) -> None:
        sampler = ILSSampler(
            ILSSamplerConfig(n_runs=5, max_iter=50, accept_equal=accept_equal, seed=SEED)
        )
        result = sampler.sample(NKLandscape(n=20, k=4, instance_seed=SEED))
        assert not result.trace_df.empty
        validate_without_warnings(result.trace_df)
