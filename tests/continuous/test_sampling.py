import numpy as np
import pytest

from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig
from tests.conftest import DOMAIN_2D, SEED, sphere


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
    def test_run_is_recorded_as_self_loop(self) -> None:
        """A run started at the optimum keeps it, although no move is accepted (issue #39)."""
        sampler = BasinHoppingSampler(BasinHoppingSamplerConfig(n_runs=3, max_iter=5, seed=SEED))
        result = sampler.sample(sphere, DOMAIN_2D, initial_points=np.zeros((3, 2)))
        assert not any(r["accepted"] for r in result.raw_records)
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
