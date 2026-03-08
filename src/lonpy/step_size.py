import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from lonpy.sampling import BasinHoppingSampler, BasinHoppingSamplerConfig


@dataclass
class StepSizeEstimatorConfig:
    """
    Configuration for step size estimation.

    Attributes:
        n_samples: Number of random initial points to evaluate. Default: `100`.
        n_perturbations: Number of perturbations per sample point. Default: `30`.
        target_escape_rate: Target escape rate to find (0.5 = 50% of perturbations escape). Default: `0.5`.
        search_precision: Decimal digits of precision for step size search.
            The algorithm refines by dividing the increment by 10 each iteration,
            so ``search_precision=4`` means 4 refinement rounds yielding resolution 0.0001. Default: `4`.
        coordinate_precision: Decimal precision for coordinate rounding and hashing.
            Solutions rounded to this precision are considered identical.
            Use `None` for full double precision (no rounding). Default: `4`.
        minimizer_method: Minimization method passed to ``scipy.optimize.minimize``. Can be a
            string or a callable implementing a custom solver.
            See `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            for the full list of supported methods and their options. Default: `"L-BFGS-B"`.
        minimizer_options: Solver-specific options passed as the ``options`` argument to
            ``scipy.optimize.minimize``. The available keys depend on the chosen
            ``minimizer_method``. Use ``None`` to rely on scipy's defaults.
            Default: `{"ftol": 1e-07, "gtol": 0, "maxiter": 15000}`.
        bounded: Whether to enforce domain bounds during perturbation. Default: `True`.
        seed: Random seed for reproducibility. Default: `None`.
    """

    n_samples: int = 100
    n_perturbations: int = 30
    target_escape_rate: float = 0.5
    search_precision: int = 4
    coordinate_precision: int | None = 4
    minimizer_method: str | Callable | None = "L-BFGS-B"
    minimizer_options: dict | None = field(
        default_factory=lambda: {"ftol": 1e-07, "gtol": 0, "maxiter": 15000}
    )
    bounded: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be greater than 0.")
        if self.n_perturbations <= 0:
            raise ValueError("n_perturbations must be greater than 0.")
        if not (0 <= self.target_escape_rate <= 1):
            raise ValueError("target_escape_rate must be between 0 and 1.")
        if self.search_precision < 1:
            raise ValueError("search_precision must be at least 1.")


@dataclass(frozen=True)
class StepSizeResult:
    """
    Result of step size estimation.

    Attributes:
        step_size: Estimated optimal step size (percentage of domain range).
        escape_rate: Achieved escape rate at this step size.
        error: Absolute difference between achieved and target escape rate.
    """

    step_size: float
    escape_rate: float
    error: float


class StepSizeEstimator:
    """
    Estimates the optimal fixed step size for basin-hopping sampling.

    The optimal step size is defined as the one that produces an escape rate
    closest to a target ``X`` (default 0.5), meaning ~ ``X * 100%`` of perturbations lead to
    a different local optimum.

    The search uses a decimal refinement approach, progressively narrowing
    the step size to the configured precision.

    Computational cost:
        `_compute_escape_rate` is called once per tested step size. Each call runs
        `n_samples` baseline minimizations and `n_samples * n_perturbations`
        perturbed minimizations (defaults: 100 + 3000 minimizations per step size).
        Since multiple step sizes are evaluated during refinement
        (`search_precision` dependent), total minimizations can become large for
        expensive objective functions. For expensive objectives, start by reducing
        `n_samples` and/or `n_perturbations`, then increase them once a reasonable
        step-size range is identified.

    Example:
        >>> import numpy as np
        >>> estimator = StepSizeEstimator()
        >>> result = estimator.estimate(problem, [(-5, 5)] * 2)
        >>> print(f"Step size: {result.step_size}, escape rate: {result.escape_rate:.3f}")
    """

    def __init__(self, config: StepSizeEstimatorConfig | None = None):
        self.config = config or StepSizeEstimatorConfig()

    def _make_sampler(self) -> BasinHoppingSampler:
        sampler_config = BasinHoppingSamplerConfig(
            coordinate_precision=self.config.coordinate_precision,
            bounded=self.config.bounded,
            minimizer_method=self.config.minimizer_method,
            minimizer_options=self.config.minimizer_options,
            step_mode="percentage",
            seed=self.config.seed,
        )
        return BasinHoppingSampler(sampler_config)

    def _compute_escape_rate(
        self,
        func: Callable[[np.ndarray], float],
        domain_array: np.ndarray,
        step_size: float,
        sampler: BasinHoppingSampler,
    ) -> float:
        """
        Compute the average escape rate for a given step size.

        Cost per call: `n_samples` + `n_samples * n_perturbations` minimizations.
        """
        bounds_array = domain_array if self.config.bounded else None
        p = step_size * np.abs(domain_array[:, 1] - domain_array[:, 0])

        escape_rates = []

        for _ in range(self.config.n_samples):
            x0 = sampler._rng.uniform(domain_array[:, 0], domain_array[:, 1])
            try:
                res = minimize(
                    func,
                    x0,
                    method=self.config.minimizer_method,
                    options=self.config.minimizer_options,
                    bounds=bounds_array if self.config.bounded else None,
                )
            except ValueError as e:
                warnings.warn(
                    f"Initial minimize failed with ValueError: {e}. "
                    f"Starting point: {x0}. Skipping run.",
                    stacklevel=3,
                )
                continue
            optimum = res.x
            optimum_rounded = sampler._round_value(optimum, self.config.coordinate_precision)
            optimum_hash = sampler._hash_solution(optimum_rounded)

            escapes = 0
            for _ in range(self.config.n_perturbations):
                x_perturbed = sampler._perturbation(optimum, p, bounds_array)
                try:
                    res_perturbed = minimize(
                        func,
                        x_perturbed,
                        method=self.config.minimizer_method,
                        options=self.config.minimizer_options,
                        bounds=bounds_array,
                    )
                except ValueError as e:
                    # L-BFGS-B can produce internal iterates that slightly
                    # violate bounds, causing approx_derivative to fail.
                    # Skip this perturbation and try the next one.
                    warnings.warn(
                        f"Minimize after perturbation "
                        f"failed with ValueError: {e}. "
                        f"Perturbed point: {x_perturbed}. Skipping perturbation.",
                        stacklevel=3,
                    )
                    continue
                new_hash = sampler._hash_solution(res_perturbed.x)
                if new_hash != optimum_hash:
                    escapes += 1

            escape_rates.append(escapes / self.config.n_perturbations)

        return float(np.mean(escape_rates))

    def estimate(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
    ) -> StepSizeResult:
        """
        Estimate the optimal step size for basin-hopping sampling.

        Args:
            func: Objective function to minimize (f: R^n -> R).
            domain: List of (lower, upper) bounds per dimension.

        Returns:
            StepSizeResult with the estimated step size, achieved escape rate, and error.
        """
        domain_array = np.array(domain)
        sampler = self._make_sampler()

        step = 0.1
        increment = 0.1
        target = self.config.target_escape_rate

        best_lower: tuple[float, float] | None = None  # (step, rate)
        best_upper: tuple[float, float] | None = None  # (step, rate)
        last_tested: tuple[float, float] | None = None

        for _ in range(self.config.search_precision):
            while step <= 1.0 + 1e-12:  # epsilon to ensure step=1.0 is tested despite float drift
                rate = self._compute_escape_rate(func, domain_array, step, sampler)
                last_tested = (step, rate)

                if rate < target:
                    if best_lower is None or abs(rate - target) < abs(best_lower[1] - target):
                        best_lower = (step, rate)
                    step += increment
                else:
                    if best_upper is None or abs(rate - target) < abs(best_upper[1] - target):
                        best_upper = (step, rate)
                    # Refine: reduce increment and resume from lower bound
                    increment /= 10
                    if best_lower is not None:
                        step = best_lower[0] + increment
                    else:
                        step = increment
                    break
            else:
                # Reached step > 1.0 without finding upper bound
                break

        # Select the candidate closest to target
        candidates = [c for c in (best_lower, best_upper, last_tested) if c is not None]
        best_step, best_rate = min(candidates, key=lambda c: abs(c[1] - target))

        return StepSizeResult(
            step_size=round(best_step, self.config.search_precision),
            escape_rate=best_rate,
            error=abs(best_rate - target),
        )
