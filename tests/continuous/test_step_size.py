from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from lonkit import StepSizeEstimator, StepSizeEstimatorConfig
from tests.conftest import DOMAIN_2D, SEED, sphere


def _res(x: float, fun: float = 0.0, success: bool = True) -> OptimizeResult:
    return OptimizeResult(x=np.full(2, x), fun=fun, nfev=1, success=success, message="")


class TestEscapeRateWithFailures:
    def test_failed_perturbations_are_excluded_from_denominator(self) -> None:
        estimator = StepSizeEstimator(
            StepSizeEstimatorConfig(n_samples=1, n_perturbations=4, seed=SEED)
        )
        # Two of four perturbations fail; one of the two evaluated escapes.
        results = [_res(0.0), ValueError("bad"), _res(1.0, float("nan")), _res(0.0), _res(1.0)]
        with (
            patch("lonkit.continuous.sampling.minimize", side_effect=results),
            pytest.warns(UserWarning, match="Skipping perturbation"),
        ):
            rate = estimator._compute_escape_rate(
                sphere, np.array(DOMAIN_2D), 0.1, estimator._make_sampler()
            )
        assert rate == 0.5

    def test_sample_without_evaluated_perturbations_is_skipped(self) -> None:
        estimator = StepSizeEstimator(
            StepSizeEstimatorConfig(n_samples=2, n_perturbations=1, seed=SEED)
        )
        results = [_res(0.0), ValueError("bad"), _res(0.0), _res(1.0)]
        with (
            patch("lonkit.continuous.sampling.minimize", side_effect=results),
            pytest.warns(UserWarning),
        ):
            rate = estimator._compute_escape_rate(
                sphere, np.array(DOMAIN_2D), 0.1, estimator._make_sampler()
            )
        assert rate == 1.0

    def test_all_samples_failing_raises(self) -> None:
        def nan_everywhere(x: np.ndarray) -> float:
            return float("nan")

        estimator = StepSizeEstimator(
            StepSizeEstimatorConfig(n_samples=3, n_perturbations=2, seed=SEED)
        )
        with (
            pytest.warns(UserWarning, match="Initial local search failed"),
            pytest.raises(RuntimeError, match="Could not estimate the escape rate"),
        ):
            estimator.estimate(nan_everywhere, DOMAIN_2D)
