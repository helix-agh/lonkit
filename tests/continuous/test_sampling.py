from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig, compute_lon
from tests.conftest import DOMAIN_2D, SEED, sphere


class TestComputeLonDomain:
    @pytest.mark.parametrize(
        "dim, lower, upper, message",
        [
            (0, -1, 1, "positive integer"),
            (-1, -1, 1, "positive integer"),
            (1.5, -1, 1, "positive integer"),
            (True, -1, 1, "positive integer"),
            (2, [0], [1], "exactly dim"),
            (2, [0, 0], [1], "exactly dim"),
            (2, [0], 1, "exactly dim"),
            (2, [[0, 0]], [[1, 1]], "exactly dim"),
            (2, -np.inf, 1, "finite"),
            (2, 0, [1, np.inf], "finite"),
            (2, [0, np.nan], 1, "finite"),
            (2, [0, 2], [1, 1], "lower_bound must be <="),
        ],
    )
    def test_invalid_domain_fails_before_sampling(self, dim, lower, upper, message):
        with patch.object(BasinHoppingSampler, "sample") as sample:
            with pytest.raises(ValueError, match=message):
                compute_lon(sphere, dim, lower, upper)
            sample.assert_not_called()

    @pytest.mark.parametrize(
        "lower, upper",
        [(-1, 1), ([-1, 0], [1, 0]), (np.array([-1, -1]), 1)],
    )
    def test_valid_domain_preserves_dimension(self, lower, upper):
        dimensions = []

        def objective(x):
            dimensions.append(len(x))
            return sphere(x)

        lon = compute_lon(
            objective,
            2,
            lower,
            upper,
            config=BasinHoppingSamplerConfig(n_runs=1, max_iter=1, seed=SEED),
        )
        assert dimensions and set(dimensions) == {2}
        assert lon.n_vertices > 0


class TestHashSolution:
    @pytest.mark.parametrize("precision", [0, 1, 2, 3])
    def test_small_positive_and_negative_values_hash_equally(self, precision: int) -> None:
        """Small values that round to zero should produce the same hash regardless of sign."""
        config = BasinHoppingSamplerConfig(coordinate_precision=precision)
        sampler = BasinHoppingSampler(config)

        threshold = 0.5 * 10 ** (-precision)
        small_pos = np.array([threshold / 2, threshold / 3])
        small_neg = -small_pos
        small_pos_rounded = sampler._round_value(small_pos, precision)
        small_neg_rounded = sampler._round_value(small_neg, precision)
        assert sampler._hash_solution(small_pos_rounded) == sampler._hash_solution(
            small_neg_rounded
        )

    def test_negative_zero_hashes_same_as_positive_zero(self) -> None:
        config = BasinHoppingSamplerConfig(coordinate_precision=1)
        sampler = BasinHoppingSampler(config)
        small_pos_rounded = sampler._round_value(np.array([0.0, 0.0]), 1)
        small_neg_rounded = sampler._round_value(np.array([-0.0, -0.0]), 1)
        assert sampler._hash_solution(small_pos_rounded) == sampler._hash_solution(
            small_neg_rounded
        )

    def test_negative_zero_hashes_same_as_positive_zero_no_precision(self) -> None:
        config = BasinHoppingSamplerConfig(coordinate_precision=None)
        sampler = BasinHoppingSampler(config)

        assert sampler._hash_solution(np.array([-0.0])) == sampler._hash_solution(np.array([0.0]))


class TestRunsWithoutAcceptedTransitions:
    @pytest.mark.parametrize("preserve", [True, False])
    def test_mixed_runs_keep_accepted_transitions(self, preserve: bool) -> None:
        sampler = BasinHoppingSampler(
            BasinHoppingSamplerConfig(
                n_runs=2, max_iter=1, seed=SEED, preserve_runs_without_transitions=preserve
            )
        )
        # Run 1 rejects its move; run 2 accepts an improvement.
        results = [
            OptimizeResult(x=np.full(2, x), fun=f, nfev=1)
            for x, f in [(0.0, 0.0), (1.0, 2.0), (2.0, 8.0), (1.0, 2.0)]
        ]
        with patch("lonkit.continuous.sampling.minimize", side_effect=results):
            result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.array([[0, 0], [2, 2]]))

        assert [r["accepted"] for r in result.raw_records] == [False, True]
        assert result.trace_df["run"].tolist() == ([1, 2] if preserve else [2])
        lon = sampler.sample_to_lon(result)
        assert lon.n_vertices == (3 if preserve else 2)
        assert lon.n_edges == 1
        edge = lon.graph.es[0]
        assert lon.vertex_names[edge.source] == "2.00000_2.00000"
        assert lon.vertex_names[edge.target] == "1.00000_1.00000"
        assert edge["Count"] == 1
        assert lon.compute_performance_metrics(known_best=0.0) == {
            "success": 0.5 if preserve else 0.0,
            "deviation": 1.0 if preserve else 2.0,
        }

    @pytest.mark.parametrize("preserve", [True, False])
    def test_all_perturbations_fail(self, preserve: bool) -> None:
        sampler = BasinHoppingSampler(
            BasinHoppingSamplerConfig(
                n_runs=1, max_iter=2, seed=SEED, preserve_runs_without_transitions=preserve
            )
        )
        initial_result = OptimizeResult(x=np.zeros(2), fun=0.0, nfev=1)
        with (
            patch(
                "lonkit.continuous.sampling.minimize",
                side_effect=[initial_result, ValueError("failed"), ValueError("failed")],
            ),
            pytest.warns(UserWarning) as warnings,
        ):
            result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((1, 2)))

        messages = [str(w.message) for w in warnings]
        assert sum("minimize after perturbation failed" in m for m in messages) == 2
        assert result.nfev == 1
        lon = sampler.sample_to_lon(result)
        if not preserve:
            assert result.raw_records == []
            assert result.trace_df.empty
            assert lon.n_vertices == 0
            return

        assert len(result.raw_records) == 1
        assert result.raw_records[0]["iteration"] == -1
        assert not result.raw_records[0]["accepted"]
        assert result.trace_df["run"].tolist() == [1]
        assert (result.trace_df["node1"] == result.trace_df["node2"]).all()
        assert lon.n_vertices == 1
        assert lon.n_edges == 0
        assert lon.final_run_values is not None
        assert lon.final_run_values.to_dict() == {1: 0.0}
        assert lon.compute_performance_metrics(known_best=0.0) == {"success": 1.0, "deviation": 0.0}

    @pytest.mark.parametrize("preserve", [True, False])
    def test_run_is_recorded_as_self_loop(self, preserve: bool) -> None:
        """A run started at the optimum keeps it, although no move is accepted (issue #39)."""
        sampler = BasinHoppingSampler(
            BasinHoppingSamplerConfig(
                n_runs=3, max_iter=5, seed=SEED, preserve_runs_without_transitions=preserve
            )
        )
        result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((3, 2)))
        assert not any(r["accepted"] for r in result.raw_records)
        if not preserve:
            assert len(result.raw_records) == 15
            assert result.trace_df.empty
            lon = sampler.sample_to_lon(result)
            assert lon.n_vertices == 0
            assert lon.final_run_values is None
            return
        assert result.trace_df["run"].tolist() == [1, 2, 3]
        assert (result.trace_df["node1"] == result.trace_df["node2"]).all()

        lon = sampler.sample_to_lon(result)
        assert lon.n_vertices == 1
        assert lon.n_edges == 0
        assert lon.compute_performance_metrics(known_best=0.0) == {"success": 1.0, "deviation": 0.0}

    def test_non_finite_initial_minimum_is_skipped(self) -> None:
        def nan_for_negative_x0(x: np.ndarray) -> float:
            return float("nan") if x[0] < 0 else 1.0

        config = BasinHoppingSamplerConfig(
            n_runs=2, max_iter=3, seed=SEED, minimizer_method="Nelder-Mead"
        )
        result = BasinHoppingSampler(config).sample(
            nan_for_negative_x0, DOMAIN_2D, initial_points=np.array([[-2.0, 0.0], [2.0, 0.0]])
        )
        assert set(result.trace_df["run"]) == {2}


def _res(x: float, fun: float, success: bool = True) -> OptimizeResult:
    return OptimizeResult(
        x=np.full(2, x),
        fun=fun,
        nfev=1,
        success=success,
        status=0 if success else 2,
        message="ok" if success else "ABNORMAL",
    )


class TestMinimizerFailures:
    def test_unconverged_finite_result_is_used(self) -> None:
        sampler = BasinHoppingSampler(BasinHoppingSamplerConfig(n_runs=1, max_iter=1, seed=SEED))
        results = [_res(2.0, 8.0), _res(1.0, 2.0, success=False)]
        with patch("lonkit.continuous.sampling.minimize", side_effect=results):
            result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((1, 2)))

        (record,) = result.raw_records
        assert record["accepted"]
        assert result.nfev == 2
        assert sampler.sample_to_lon(result).n_vertices == 2

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_initial_minimum_fails_run(self, bad: float) -> None:
        def f(x: np.ndarray) -> float:
            return bad if x[0] > 3 else float(np.sum(x**2))

        sampler = BasinHoppingSampler(
            BasinHoppingSamplerConfig(n_runs=4, n_iter_no_change=5, seed=1)
        )
        points = np.array([[4.0, 1.0], [5.0, 2.0], [-1.0, -1.0], [-2.0, 1.0]])
        with pytest.warns(UserWarning, match="initial local search failed"):
            result = sampler.sample(f, [(-5, 5)] * 2, initial_points=points)

        assert set(result.trace_df["run"]) == {3, 4}
        assert np.isfinite(result.trace_df[["fit1", "fit2"]].to_numpy()).all()
        assert all(r["run"] in (3, 4) for r in result.raw_records)
        sampler.sample_to_lon(result)  # trace is valid

    def test_non_finite_mid_run_is_never_accepted(self) -> None:
        sampler = BasinHoppingSampler(BasinHoppingSamplerConfig(n_runs=1, max_iter=2, seed=SEED))
        results = [_res(2.0, 8.0), _res(1.0, float("-inf")), _res(1.0, 2.0)]
        with (
            patch("lonkit.continuous.sampling.minimize", side_effect=results),
            pytest.warns(UserWarning, match="minimize after perturbation failed"),
        ):
            result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((1, 2)))

        assert [r["accepted"] for r in result.raw_records] == [False, True]
        assert result.trace_df[["fit1", "fit2"]].to_numpy().tolist() == [[8.0, 2.0]]

    def test_initial_value_error_reports_failed_run(self) -> None:
        sampler = BasinHoppingSampler(BasinHoppingSamplerConfig(n_runs=2, max_iter=1, seed=SEED))
        results = [ValueError("bad"), _res(2.0, 8.0), _res(1.0, 2.0)]
        with (
            patch("lonkit.continuous.sampling.minimize", side_effect=results),
            pytest.warns(UserWarning, match="Run 1: initial local search failed"),
        ):
            result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((2, 2)))
        assert result.trace_df["run"].tolist() == [2]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_failed_perturbations_stop_after_no_change_limit(self, bad: float) -> None:
        sampler = BasinHoppingSampler(
            BasinHoppingSamplerConfig(n_runs=1, n_iter_no_change=2, max_iter=None, seed=SEED)
        )
        results = [_res(2.0, 8.0), _res(1.0, bad), _res(1.0, bad)]
        with (
            patch("lonkit.continuous.sampling.minimize", side_effect=results) as minimize,
            pytest.warns(UserWarning, match="minimize after perturbation failed") as warnings,
        ):
            result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((1, 2)))

        assert minimize.call_count == 3
        assert len(warnings) == 2
        assert not any(r["accepted"] for r in result.raw_records)
        assert result.nfev == 3
        assert result.trace_df["fit1"].tolist() == [8.0]
        assert sampler.sample_to_lon(result).n_vertices == 1

    def test_failed_runs_match_across_n_jobs(self) -> None:
        def f(x: np.ndarray) -> float:
            return float("nan") if x[0] > 3 else float(np.sum(x**2))

        results = []
        messages = []
        points = np.array([[4.0, 1.0], [-1.0, -1.0]])
        for n_jobs in (1, 2):
            config = BasinHoppingSamplerConfig(
                n_runs=2, n_iter_no_change=3, seed=SEED, n_jobs=n_jobs
            )
            with pytest.warns(UserWarning) as caught:
                results.append(
                    BasinHoppingSampler(config).sample(f, DOMAIN_2D, initial_points=points)
                )
            messages.append(sorted(str(w.message) for w in caught))
        assert any("Run 1: initial local search failed" in m for m in messages[1])
        assert messages[0] == messages[1]
        pd.testing.assert_frame_equal(results[0].trace_df, results[1].trace_df)
        assert results[0].nfev == results[1].nfev
        assert set(results[0].trace_df["run"]) == {2}
