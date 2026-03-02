import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lonpy.lon import LON, LONConfig

StepMode = Literal["percentage", "fixed"]


@dataclass
class BasinHoppingSamplerConfig:
    """
    Configuration for Basin-Hopping sampling.

    Attributes:
        n_runs: Number of independent Basin-Hopping runs.  Default: `100`.
        n_iter_no_change: Maximum number of consecutive non-improving perturbations before stopping each run.
            Use `None` for no limit. Setting both `n_iter_no_change` and `max_iter` to `None` will result in an error. Default: `1000`.
        max_iter: Optional maximum number of total iterations (perturbation steps) per run.
            Use `None` for no limit. Setting both `n_iter_no_change` and `max_iter` to `None` will result in an error. Default: `None`.
        step_mode: Perturbation mode - `"percentage"` (of domain range)
            or `"fixed"` (absolute step size). Default: `"fixed"`.
        step_size: Perturbation magnitude (interpretation depends on step_mode). Default: `0.01`.
        fitness_precision: Decimal precision for fitness values.
            Use `None` for full double precision. Passing negative values behaves the same as passing `None`. Default: `None`.
        coordinate_precision: Decimal precision for coordinate rounding and hashing.
            Solutions rounded to this precision are considered identical.
            Use `None` for full double precision (no rounding). Passing negative values behaves the same as passing `None`. Default: `5`.
        bounded: Whether to enforce domain bounds during perturbation. Default: `True`.
        minimizer_method: Minimization method passed to ``scipy.optimize.minimize``. Can be a
            string or a callable implementing a custom solver.
            See `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            for the full list of supported methods and their options. Default: `"L-BFGS-B"`.
        minimizer_options: Solver-specific options passed as the ``options`` argument to
            ``scipy.optimize.minimize``. The available keys depend on the chosen
            ``minimizer_method``. Use ``None`` to rely on scipy's defaults.
            Default: `{"ftol": 1e-07, "gtol": 0, "maxiter": 15000}`.
        seed: Random seed for reproducibility. Default: `None`.
    """

    n_runs: int = 100
    n_iter_no_change: int | None = 1000
    max_iter: int | None = None
    step_mode: StepMode = "fixed"
    step_size: float = 0.01
    fitness_precision: int | None = None
    coordinate_precision: int | None = 5
    bounded: bool = True
    minimizer_method: str | Callable | None = "L-BFGS-B"
    minimizer_options: dict | None = field(
        default_factory=lambda: {"ftol": 1e-07, "gtol": 0, "maxiter": 15000}
    )
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_iter_no_change is not None and self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive or None.")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be positive or None.")
        if self.n_iter_no_change is None and self.max_iter is None:
            raise ValueError(
                "At least one stopping criterion must be set: n_iter_no_change and/or max_iter."
            )


class BasinHoppingSampler:
    """
    Basin-Hopping sampler for constructing Local Optima Networks.

    Basin-Hopping is a global optimization algorithm that combines random
    perturbations with local minimization. This implementation records
    transitions between local optima for LON construction.

    Example:
        >>> config = BasinHoppingSamplerConfig(n_runs=10, n_iter_no_change=1000)
        >>> sampler = BasinHoppingSampler(config)
        >>> lon = sampler.sample_to_lon(objective_func, domain)
    """

    def __init__(self, config: BasinHoppingSamplerConfig | None = None):
        self.config = config or BasinHoppingSamplerConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def _perturbation(
        self,
        x: np.ndarray,
        p: np.ndarray,
        bounds: np.ndarray | None = None,
    ) -> np.ndarray:
        y = x + self._rng.uniform(low=-p, high=p)
        if self.config.bounded and bounds is not None:
            return np.clip(y, bounds[:, 0], bounds[:, 1])
        return y

    def _round_value(self, value: np.ndarray, precision: int | None) -> np.ndarray:
        if precision is None or precision < 0:
            return value
        return np.round(value, precision)

    def _hash_solution(self, x: np.ndarray) -> str:
        """
        Create hash string for a solution.

        Creates a unique identifier for a local optimum based on
        rounded coordinates.

        Args:
            x: Solution coordinates.

        Returns:
            Hash string identifying the local optimum.
        """

        x = (
            x + 0.0
        )  # Convert -0.0 to 0.0 for consistent hashing (avoids in-place mutation of input)

        precision = self.config.coordinate_precision
        formatter = str if precision is None or precision < 0 else lambda v: f"{v:.{precision}f}"
        hash_str = "_".join(formatter(v) for v in x)

        return hash_str

    def _basin_hopping_sampling(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
        initial_points: np.ndarray,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """
        Run Basin-Hopping sampling to generate LON data.

        Args:
            func: Objective function to minimize (f: R^n_var -> R).
            domain: List of (lower, upper) bounds per dimension.
            initial_points: Array of shape (config.n_runs, n_var) with initial points.
            progress_callback: Optional callback(run, total_runs) for progress.

        Returns:
            List of raw sampling records, one per perturbation step.
            Each record is a dict with keys: run, iteration, current_x,
            current_f, new_x, new_f, accepted.
        """
        n_var = len(domain)
        domain_array = np.array(domain)

        # Compute step size based on mode
        if self.config.step_mode == "percentage":
            p = self.config.step_size * np.abs(domain_array[:, 1] - domain_array[:, 0])
        else:
            p = self.config.step_size * np.ones(n_var)

        bounds_array = domain_array if self.config.bounded else None
        raw_records = []

        for run in range(1, self.config.n_runs + 1):
            if progress_callback:
                progress_callback(run, self.config.n_runs)

            try:
                res = minimize(
                    func,
                    initial_points[run - 1],
                    method=self.config.minimizer_method,
                    options=self.config.minimizer_options,
                    bounds=bounds_array if self.config.bounded else None,
                )
            except ValueError as e:
                warnings.warn(
                    f"Run {run}: initial minimize failed with ValueError: {e}. "
                    f"Starting point: {initial_points[run - 1]}. Skipping run.",
                    stacklevel=3,
                )
                continue

            current_x = res.x
            current_f = res.fun

            iters_without_improvement = 0
            iter_index = 0

            while True:
                if self.config.max_iter is not None and iter_index >= self.config.max_iter:
                    break
                if (
                    self.config.n_iter_no_change is not None
                    and iters_without_improvement >= self.config.n_iter_no_change
                ):
                    break

                x_perturbed = self._perturbation(current_x, p, bounds_array)
                try:
                    res = minimize(
                        func,
                        x_perturbed,
                        method=self.config.minimizer_method,
                        options=self.config.minimizer_options,
                        bounds=bounds_array if self.config.bounded else None,
                    )
                except ValueError as e:
                    # L-BFGS-B can produce internal iterates that slightly
                    # violate bounds, causing approx_derivative to fail.
                    # Skip this perturbation and try the next one.
                    warnings.warn(
                        f"Run {run}, iteration {iter_index}: minimize after perturbation "
                        f"failed with ValueError: {e}. "
                        f"Perturbed point: {x_perturbed}. Skipping perturbation.",
                        stacklevel=3,
                    )
                    iters_without_improvement += 1
                    iter_index += 1
                    continue

                new_x = res.x
                new_f = res.fun

                raw_records.append(
                    {
                        "run": run,
                        "iteration": iter_index,
                        "current_x": current_x.copy(),
                        "current_f": current_f,
                        "new_x": new_x.copy(),
                        "new_f": new_f,
                        "accepted": new_f <= current_f,
                    }
                )

                if self.config.n_iter_no_change is not None:
                    if new_f < current_f:
                        iters_without_improvement = 0
                    else:
                        iters_without_improvement += 1

                # Acceptance criterion (minimization: accept if better or equal)
                if new_f <= current_f:
                    current_x = new_x.copy()
                    current_f = new_f

                iter_index += 1

        return raw_records

    def _construct_trace_data(self, raw_records: list[dict]) -> pd.DataFrame:
        """
        Construct trace data from accepted transitions in raw records.

        Args:
            raw_records: List of raw sampling records from basin hopping.

        Returns:
            DataFrame with columns `[run, fit1, node1, fit2, node2]` representing
            actual transitions from current_x to new_x for each accepted move.
        """
        trace_records = []

        for rec in raw_records:
            if not rec["accepted"]:
                continue

            from_x = rec["current_x"]
            from_f = rec["current_f"]
            to_x = rec["new_x"]
            to_f = rec["new_f"]

            from_x_rounded = self._round_value(from_x, self.config.coordinate_precision)
            to_x_rounded = self._round_value(to_x, self.config.coordinate_precision)

            node1 = self._hash_solution(from_x_rounded)
            node2 = self._hash_solution(to_x_rounded)

            fit1 = self._round_value(from_f, self.config.fitness_precision)
            fit2 = self._round_value(to_f, self.config.fitness_precision)

            trace_records.append(
                {
                    "run": rec["run"],
                    "fit1": fit1,
                    "node1": node1,
                    "fit2": fit2,
                    "node2": node2,
                }
            )

        trace_df = pd.DataFrame(trace_records, columns=["run", "fit1", "node1", "fit2", "node2"])
        return trace_df

    def _resolve_initial_points(
        self,
        initial_points: np.ndarray | None,
        domain: list[tuple[float, float]],
    ) -> np.ndarray:
        n_runs = self.config.n_runs
        n_var = len(domain)
        domain_array = np.array(domain)

        if initial_points is None:
            return self._rng.uniform(domain_array[:, 0], domain_array[:, 1], size=(n_runs, n_var))

        initial_points = np.asarray(initial_points, dtype=float)

        if initial_points.ndim != 2 or initial_points.shape[1] != n_var:
            raise ValueError(
                f"initial_points must have shape (n_runs, {n_var}), got {initial_points.shape}."
            )

        if initial_points.shape[0] != n_runs:
            raise ValueError(
                f"initial_points has {initial_points.shape[0]} points, "
                f"but n_runs is {n_runs}. "
                f"These must match."
            )

        if self.config.bounded:
            lower = domain_array[:, 0]
            upper = domain_array[:, 1]
            if np.any(initial_points < lower) or np.any(initial_points > upper):
                raise ValueError(
                    "initial_points contains values outside the domain bounds. "
                    "All points must satisfy lower_bound <= x <= upper_bound "
                    "when bounded=True."
                )

        return initial_points

    def sample(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
        initial_points: np.ndarray | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Run Basin-Hopping sampling and construct trace data.

        Args:
            func: Objective function to minimize (f: R^n_var -> R).
            domain: List of (lower, upper) bounds per dimension.
            initial_points: Optional array of shape (`config.n_runs`, `n_var`) with
                starting points for each run. If `None`, points are sampled
                uniformly at random from the domain. Default: `None`.
            progress_callback: Optional callback(run, total_runs) for progress. Default: `None`.

        Returns:
            Tuple of (trace_df, raw_records):
                - trace_df: DataFrame with columns `[run, fit1, node1, fit2, node2]`
                - raw_records: List of dicts with detailed iteration data.
        """
        resolved_points = self._resolve_initial_points(initial_points, domain)

        # Collect all raw sampling data
        raw_records = self._basin_hopping_sampling(func, domain, resolved_points, progress_callback)

        # Construct trace data from accepted transitions
        trace_df = self._construct_trace_data(raw_records)

        return trace_df, raw_records

    def sample_to_lon(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
        initial_points: np.ndarray | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        lon_config: LONConfig | None = None,
    ) -> LON:
        """
        Run Basin-Hopping sampling and construct a LON.

        Convenience wrapper that calls `sample()` and passes the resulting
        trace data to `LON.from_trace_data()`. Equivalent to calling
        `sample()` followed by `LON.from_trace_data(trace_df, config=lon_config)`.

        Args:
            func: Objective function to minimize (f: R^n_var -> R).
            domain: List of (lower, upper) bounds per dimension.
            initial_points: Optional array of shape (`config.n_runs`, `n_var`) with
                starting points for each run. If `None`, points are sampled
                uniformly at random from the domain. Default: `None`.
            progress_callback: Optional callback(run, total_runs) for progress. Default: `None`.
            lon_config: LON construction configuration. If `None`, uses default
                `LONConfig`. Default: `None`.

        Returns:
            `LON` instance constructed from the sampling trace.
        """
        trace_df, _ = self.sample(func, domain, initial_points, progress_callback)

        if trace_df.empty:
            return LON()

        _lon_config = replace(lon_config) if lon_config is not None else LONConfig()

        if _lon_config.eq_atol is None:
            p = self.config.fitness_precision
            if p is not None and p >= 0:
                _lon_config.eq_atol = 10 ** -(p + 1)

        return LON.from_trace_data(trace_df, config=_lon_config)


def compute_lon(
    func: Callable[[np.ndarray], float],
    dim: int,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    initial_points: np.ndarray | None = None,
    config: BasinHoppingSamplerConfig | None = None,
    lon_config: LONConfig | None = None,
) -> LON:
    """
    Compute a LON from an objective function.

    This is the simplest way to construct a Local Optima Network.
    For more control, use BasinHoppingSampler directly.

    Args:
        func: Objective function f(x) -> float to minimize, where x is in R^n_var.
        dim: Number of dimensions (n_var).
        lower_bound: Lower bound (scalar or per-dimension list/array).
        upper_bound: Upper bound (scalar or per-dimension list/array).
        initial_points: Optional array of shape (`config.n_runs`, `dim`) with starting
            points for each run. If `None`, points are sampled uniformly at
            random from the domain. Default: `None`.
        config: Basin-Hopping sampler configuration. Uses default
            BasinHoppingSamplerConfig if not provided.
        lon_config: LON construction configuration. Uses default
            LONConfig if not provided.

    Returns:
        LON instance.

    Example:
        >>> import numpy as np
        >>> def sphere(x):
        ...     return np.sum(x**2)
        >>> lon = compute_lon(sphere, dim=5, lower_bound=-5.0, upper_bound=5.0)
        >>> print(f"Found {lon.n_vertices} local optima")
    """
    # Convert scalars to lists, leave sequences as-is
    lower_bounds: Sequence[float] = (
        [lower_bound] * dim if isinstance(lower_bound, int | float) else lower_bound
    )
    upper_bounds: Sequence[float] = (
        [upper_bound] * dim if isinstance(upper_bound, int | float) else upper_bound
    )

    domain = list(zip(lower_bounds, upper_bounds, strict=True))

    sampler = BasinHoppingSampler(config)
    return sampler.sample_to_lon(func, domain, initial_points=initial_points, lon_config=lon_config)
