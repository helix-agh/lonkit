from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lonkit.discrete.problems.problem import DiscreteProblem
from lonkit.lon import LON, LONConfig


@dataclass
class ILSSamplerConfig:
    """
    Configuration for Iterated Local Search sampling.

    Attributes:
        n_runs: Number of independent ILS runs. Default: 100.
        n_iter_no_change: Maximum consecutive non-improving iterations
            before stopping each run. Use None for no limit.
            Setting both n_iter_no_change and max_iter to None will
            result in an error. Default: 100.
        max_iter: Maximum total iterations per run. Use None for no
            limit. Setting both n_iter_no_change and max_iter to None
            will result in an error. Default: None.
        accept_equal: If True, accept moves to equal-fitness optima
            (greedy with equal acceptance). If False, only accept
            strictly improving moves. Default: True.
        seed: Random seed for reproducibility. Controls ALL search
            randomness: initial solution generation, local search
            scan order, and perturbation. The problem instance is
            stateless and receives the sampler's RNG. Default: None.
    """

    n_runs: int = 100
    n_iter_no_change: int | None = 100
    max_iter: int | None = None
    accept_equal: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_iter_no_change is not None and self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive or None.")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be positive or None.")
        if self.n_iter_no_change is None and self.max_iter is None:
            raise ValueError(
                "At least one stopping criterion must be set: " "n_iter_no_change and/or max_iter."
            )


@dataclass
class ILSResult:
    """
    Result of ILS sampling.

    Attributes:
        trace_df: DataFrame with columns [run, fit1, node1, fit2, node2]
            representing transitions between local optima.
        raw_records: List of dicts with per-iteration data. Each dict
            has keys: run, iteration, current_id, current_fitness,
            new_id, new_fitness, accepted.
    """

    trace_df: pd.DataFrame
    raw_records: list[dict]


class ILSSampler:
    """
    Iterated Local Search sampler for constructing Local Optima Networks
    from discrete optimization problems.

    ILS alternates between perturbation and local search to explore the
    space of local optima. Transitions between local optima are recorded
    in the same trace format as BasinHoppingSampler, enabling direct use
    with LON.from_trace_data().

    The sampler owns all search randomness via a single `numpy.random.Generator`
    instance created from `ILSSamplerConfig.seed`. The problem instance
    is stateless — the sampler passes its RNG into every problem method
    call (`random_solution`, `local_search`, `perturb`).

    Example:
        >>> from lonkit.discrete import NumberPartitioning, ILSSampler, ILSSamplerConfig
        >>> problem = NumberPartitioning(n=20, k=0.5, instance_seed=1)
        >>> config = ILSSamplerConfig(n_runs=10, n_iter_no_change=100, seed=42)
        >>> sampler = ILSSampler(config)
        >>> result = sampler.sample(problem)
        >>> lon = sampler.sample_to_lon(result)
    """

    def __init__(self, config: ILSSamplerConfig | None = None):
        self.config = config or ILSSamplerConfig()

    def sample(
        self,
        problem: DiscreteProblem[Any],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ILSResult:
        """
        Run ILS sampling and construct trace data.

        Args:
            problem: A DiscreteProblem instance defining the optimization
                problem, local search, and perturbation operators.
                The problem must be stateless — the sampler provides
                the RNG.
            progress_callback: Optional callback(run, total_runs) for
                progress reporting. Default: None.

        Returns:
            ILSResult with trace DataFrame and raw records.
        """
        rng = np.random.default_rng(self.config.seed)
        raw_records = []

        for run in range(1, self.config.n_runs + 1):
            if progress_callback:
                progress_callback(run, self.config.n_runs)

            run_records = self._ils_run(problem, run, rng)
            raw_records.extend(run_records)

        trace_df = self._construct_trace_data(raw_records)
        return ILSResult(trace_df=trace_df, raw_records=raw_records)

    def _ils_run(
        self,
        problem: DiscreteProblem[Any],
        run: int,
        rng: np.random.Generator,
    ) -> list[dict]:
        """
        Execute a single ILS run.

        The algorithm:
        1. Generate random initial solution
        2. Run local search to find first local optimum
        3. Loop:
           a. Perturb current best local optimum
           b. Run local search on perturbed solution
           c. Record transition (current -> new)
           d. Accept if new is better or equal (configurable)
           e. Stop after n_iter_no_change or max_iter

        Args:
            problem: The discrete problem instance.
            run: Run number (1-indexed, for trace DataFrame).
            rng: Random number generator (owned by the sampler).

        Returns:
            List of raw record dicts for this run.
        """
        records = []

        # Initial solution and local search
        initial_sol = problem.random_solution(rng)
        current_sol, current_fitness = problem.local_search(initial_sol, rng)
        current_id = problem.solution_id(current_sol)

        iters_without_improvement = 0
        iter_index = 0

        while True:
            # Check stopping criteria
            if self.config.max_iter is not None and iter_index >= self.config.max_iter:
                break
            if (
                self.config.n_iter_no_change is not None
                and iters_without_improvement >= self.config.n_iter_no_change
            ):
                break

            # Perturb and local search
            perturbed_sol = problem.perturb(current_sol, rng)
            new_sol, new_fitness = problem.local_search(perturbed_sol, rng)
            new_id = problem.solution_id(new_sol)

            if self.config.accept_equal:
                accepted = problem.is_better_or_equal(new_fitness, current_fitness)
            else:
                accepted = problem.is_better(new_fitness, current_fitness)

            records.append(
                {
                    "run": run,
                    "iteration": iter_index,
                    "current_id": current_id,
                    "current_fitness": current_fitness,
                    "new_id": new_id,
                    "new_fitness": new_fitness,
                    "accepted": accepted,
                }
            )

            if problem.is_better(new_fitness, current_fitness):
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

            if accepted:
                current_sol = new_sol
                current_fitness = new_fitness
                current_id = new_id

            iter_index += 1

        return records

    def _construct_trace_data(self, raw_records: list[dict]) -> pd.DataFrame:
        """
        Construct trace data from accepted transitions in raw records.

        Args:
            raw_records: List of raw sampling records from ILS.

        Returns:
            DataFrame with columns `[run, fit1, node1, fit2, node2]` representing
            accepted transitions only.
        """
        trace_records = []

        for rec in raw_records:
            if not rec["accepted"]:
                continue

            trace_records.append(
                {
                    "run": rec["run"],
                    "fit1": rec["current_fitness"],
                    "node1": rec["current_id"],
                    "fit2": rec["new_fitness"],
                    "node2": rec["new_id"],
                }
            )

        return pd.DataFrame(
            trace_records,
            columns=["run", "fit1", "node1", "fit2", "node2"],
        )

    def sample_to_lon(
        self,
        sampler_result: ILSResult,
        lon_config: LONConfig | None = None,
    ) -> LON:
        """
        Construct a LON from an `ILSResult`.

        Convenience wrapper that passes the trace data to
        LON.from_trace_data(). Equivalent to calling
        LON.from_trace_data(sampler_result.trace_df, config=lon_config).

        Args:
            sampler_result: Result returned by `sample()`.
            lon_config: LON construction configuration. If `None`, uses
                default `LONConfig`. Default: `None`.

        Returns:
            `LON` instance constructed from the sampling trace.
        """
        trace_df = sampler_result.trace_df

        if trace_df.empty:
            return LON()

        return LON.from_trace_data(trace_df, config=lon_config or LONConfig())
