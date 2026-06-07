import math
import warnings
from typing import Literal

import numpy as np

from lonkit.discrete.problems.problem import DiscreteProblem


class BitstringProblem(DiscreteProblem[list[int]]):
    """
    Base class for problems with binary string representation.

    Provides common bitstring operations:
    - Random solution generation (random bitstring)
    - Perturbation (k random bit flips)
    - Solution identity (join bits as string)
    - Hill climbing with first/best-improvement flip neighborhood

    Subclasses must implement `evaluate()` and may override
    `minimize` or `local_search()`.

    The problem is **stateless**: `n`, `n_perturbation_flips`, and
    `first_improvement` are immutable configuration. All randomness
    comes from the `rng` parameter passed by the caller.
    """

    def __init__(
        self,
        n: int,
        n_perturbation_flips: int = 2,
        first_improvement: bool = True,
    ):
        """
        Args:
            n: Length of the bitstring. Must be > 0.
            n_perturbation_flips: Number of random bit flips per perturbation.
                Must be in [1, n].
            first_improvement: If True, local search uses first-improvement
                hill climbing (stochastic -- scan order randomized each pass).
                If False, uses best-improvement (deterministic).

        Raises:
            ValueError: If n <= 0 or n_perturbation_flips is out of [1, n].
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n_perturbation_flips <= 0 or n_perturbation_flips > n:
            raise ValueError(
                f"n_perturbation_flips must be in [1, {n}], got {n_perturbation_flips}"
            )
        self.n = n
        self.n_perturbation_flips = n_perturbation_flips
        self.first_improvement = first_improvement

    def random_solution(self, rng: np.random.Generator) -> list[int]:
        result: list[int] = rng.integers(0, 2, size=self.n).tolist()
        return result

    def solution_id(self, solution: list[int]) -> str:
        return "".join(str(b) for b in solution)

    def perturb(self, solution: list[int], rng: np.random.Generator) -> list[int]:
        """Flip `n_perturbation_flips` random distinct bits."""
        sol = list(solution)  # copy
        indices = rng.choice(self.n, size=self.n_perturbation_flips, replace=False)
        for i in indices:
            sol[i] = 1 - sol[i]
        return sol

    def delta_evaluate(
        self,
        solution: list[int],  # noqa: ARG002
        index: int,  # noqa: ARG002
    ) -> float | None:
        """
        Optional delta evaluation hook for flipping bit at `index`.

        Returns the fitness *change* (delta) if efficient delta evaluation
        is supported, or None to fall back to full evaluation.

        The default implementation returns None. Subclasses like OneMax
        can override this for O(1) evaluation instead of O(n).

        Args:
            solution: Current solution (not modified).
            index: Bit index that would be flipped.

        Returns:
            Fitness delta (new_fitness - current_fitness), or None.
        """
        return None

    def local_search(
        self, solution: list[int], rng: np.random.Generator
    ) -> tuple[list[int], float]:
        """
        First/best improvement hill climbing with 1-bit-flip neighborhood.

        Scans all N bit positions. For first-improvement, the scan order
        is randomized each pass (stochastic). For best-improvement, the
        scan is deterministic. Stops when no improving neighbor exists.

        Uses `delta_evaluate()` when available for O(1) neighbor evaluation;
        falls back to full `evaluate()` otherwise.
        """
        sol = list(solution)  # copy
        current_fitness = self.evaluate(sol)
        improved = True

        while improved:
            improved = False
            indices = list(range(self.n))
            if self.first_improvement:
                rng.shuffle(indices)

            best_delta_index = -1
            best_delta_fitness = current_fitness

            for i in indices:
                # Try delta evaluation first
                delta = self.delta_evaluate(sol, i)
                if delta is not None:
                    new_fitness = current_fitness + delta
                else:
                    # Full evaluation: flip, evaluate, undo
                    sol[i] = 1 - sol[i]
                    new_fitness = self.evaluate(sol)
                    sol[i] = 1 - sol[i]

                if self.is_better(new_fitness, current_fitness):
                    if self.first_improvement:
                        # Accept immediately
                        sol[i] = 1 - sol[i]
                        current_fitness = new_fitness
                        improved = True
                        break
                    else:
                        # Track best
                        if self.is_better(new_fitness, best_delta_fitness):
                            best_delta_fitness = new_fitness
                            best_delta_index = i

            if not self.first_improvement and best_delta_index >= 0:
                sol[best_delta_index] = 1 - sol[best_delta_index]
                current_fitness = best_delta_fitness
                improved = True

        return sol, current_fitness


class NumberPartitioning(BitstringProblem):
    """
    Number Partitioning Problem (NPP).

    Given a set of N positive integers, partition them into two subsets
    such that the absolute difference of their sums is minimized.

    A solution is a bitstring of length N. Bit i=0 means item i goes
    to subset A, bit i=1 means subset B.

    Fitness = |sum(A) - sum(B)|  (minimization, optimal = 0).

    Construction: provide either explicit `weights` or both `k` and
    `instance_seed` for random instance generation.

    Args:
        n: Number of items. Must be > 0.
        k: Hardness parameter. Items drawn uniformly from [1, 2^(n*k)].
            Higher k = harder instances (phase transition around k ~ 1.0).
            Required if `weights` is not provided. Must be > 0.
        instance_seed: Seed for generating item weights.
            Required if `weights` is not provided.
        weights: Explicit item weights. If provided, `k` and
            `instance_seed` are ignored. Length must equal `n`.
        n_perturbation_flips: Number of random flips per perturbation (default: 2).
        first_improvement: If True, local search uses first-improvement
            hill climbing (stochastic -- scan order randomized each pass).
            If False, uses best-improvement (deterministic). Default: True.
    """

    @property
    def minimize(self) -> bool:
        return True

    def __init__(
        self,
        n: int,
        k: float | None = None,
        instance_seed: int | None = None,
        weights: list[int] | None = None,
        n_perturbation_flips: int = 2,
        first_improvement: bool = True,
    ):
        super().__init__(n, n_perturbation_flips, first_improvement)
        if weights is not None:
            if k is not None or instance_seed is not None:
                warnings.warn(
                    "Both `weights` and `k`/`instance_seed` were provided. "
                    "`weights` will be used and `k`/`instance_seed` will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if len(weights) != n:
                raise ValueError(f"weights length ({len(weights)}) must equal n ({n})")
            if any(w <= 0 for w in weights):
                raise ValueError("All weights must be positive")
            self.weights = list(weights)
            self.k = None
            self.instance_seed = None
        elif k is not None and instance_seed is not None:
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            self.k = k
            self.instance_seed = instance_seed
            # Generate item weights using a separate RNG
            rng = np.random.default_rng(instance_seed)
            upper = round(math.pow(2, n * k))
            self.weights = rng.integers(1, upper + 1, size=n).tolist()
        else:
            raise ValueError("Provide either `weights` or both `k` and `instance_seed`")

    def evaluate(self, solution: list[int]) -> float:
        cost_a = sum(self.weights[i] for i in range(self.n) if solution[i] == 0)
        cost_b = sum(self.weights[i] for i in range(self.n) if solution[i] == 1)
        return float(abs(cost_a - cost_b))


class NKLandscape(BitstringProblem):
    """
    Kauffman's NK Landscape.

    A tunable family of rugged fitness landscapes over bitstrings of length
    ``N``. Each of the ``N`` positions contributes a fitness component that
    depends on its own bit plus ``K`` other ("epistatic") bits. The overall
    fitness is the average of the ``N`` contributions::

        fitness(x) = (1 / N) * sum_i  f_i(x_i, x_{neighbors(i)})

    Each component ``f_i`` is defined by a lookup table of ``2^(K+1)`` values
    drawn uniformly from ``[0, 1)`` and indexed by the ``(K + 1)``-bit pattern
    formed by bit ``i`` followed by its ``K`` neighbors.

    ``K`` tunes the ruggedness:
    - ``K = 0`` gives a smooth, single-optimum landscape (each bit independent).
    - ``K = N - 1`` gives a maximally rugged, random landscape.

    This is a **maximization** problem (optimal fitness close to 1.0).

    The instance (neighbor structure and contribution tables) is fixed at
    construction time from ``instance_seed``, so two problems built with the
    same parameters are identical.

    Args:
        n: Length of the bitstring. Must be > 0.
        k: Number of epistatic interactions per position. Must be in [0, n-1].
        instance_seed: Seed for generating the neighbor structure (random model)
            and the random contribution tables. Required for reproducibility.
        neighbor_model: ``"adjacent"`` (each position interacts with its ``K``
            cyclically-following neighbors) or ``"random"`` (``K`` distinct
            positions drawn uniformly at random, excluding the position itself).
            Default: ``"adjacent"``.
        n_perturbation_flips: Number of random flips per perturbation (default: 2).
        first_improvement: If True, local search uses first-improvement
            hill climbing (stochastic -- scan order randomized each pass).
            If False, uses best-improvement (deterministic). Default: True.
    """

    @property
    def minimize(self) -> bool:
        return False

    def __init__(
        self,
        n: int,
        k: int,
        instance_seed: int,
        neighbor_model: Literal["adjacent", "random"] = "adjacent",
        n_perturbation_flips: int = 2,
        first_improvement: bool = True,
    ):
        super().__init__(n, n_perturbation_flips, first_improvement)
        if instance_seed is None:
            raise ValueError("instance_seed is required for NKLandscape")
        if k < 0 or k > n - 1:
            raise ValueError(f"k must be in [0, {n - 1}], got {k}")
        if neighbor_model not in ("adjacent", "random"):
            raise ValueError(
                f"neighbor_model must be 'adjacent' or 'random', got {neighbor_model!r}"
            )
        self.k = k
        self.instance_seed = instance_seed
        self.neighbor_model = neighbor_model

        rng = np.random.default_rng(instance_seed)
        self.neighbors = self._build_neighbors(rng)
        # Contribution tables: one row of 2^(k+1) random values per position.
        self.tables = rng.random(size=(n, 1 << (k + 1)))
        # For each bit, which positions' contributions depend on it (for delta).
        self._affected: list[list[int]] = [[] for _ in range(n)]
        for pos in range(n):
            self._affected[pos].append(pos)
            for j in self.neighbors[pos]:
                self._affected[j].append(pos)

    def _build_neighbors(self, rng: np.random.Generator) -> list[list[int]]:
        """Return the list of K epistatic neighbors for each position."""
        if self.neighbor_model == "adjacent":
            return [
                [(i + offset) % self.n for offset in range(1, self.k + 1)] for i in range(self.n)
            ]
        # random model: K distinct positions, excluding i itself
        neighbors: list[list[int]] = []
        for i in range(self.n):
            candidates = [j for j in range(self.n) if j != i]
            chosen = rng.choice(candidates, size=self.k, replace=False)
            neighbors.append(sorted(int(j) for j in chosen))
        return neighbors

    def _contribution_index(self, solution: list[int], pos: int) -> int:
        """Build the table index from bit `pos` followed by its neighbors."""
        idx = solution[pos]
        for j in self.neighbors[pos]:
            idx = (idx << 1) | solution[j]
        return idx

    def evaluate(self, solution: list[int]) -> float:
        total = 0.0
        for pos in range(self.n):
            total += self.tables[pos][self._contribution_index(solution, pos)]
        return total / self.n

    def delta_evaluate(self, solution: list[int], index: int) -> float | None:
        """
        Delta evaluation: flipping bit `index` only changes the
        contributions of positions that depend on it.
        """
        old_total = sum(
            float(self.tables[pos][self._contribution_index(solution, pos)])
            for pos in self._affected[index]
        )
        solution[index] = 1 - solution[index]
        try:
            new_total = sum(
                float(self.tables[pos][self._contribution_index(solution, pos)])
                for pos in self._affected[index]
            )
        finally:
            solution[index] = 1 - solution[index]
        return (new_total - old_total) / self.n


class OneMax(BitstringProblem):
    """
    OneMax problem: maximize the number of 1-bits.

    Fitness = sum(bits). Single global optimum at all-ones.

    Supports O(1) delta evaluation: flipping bit i changes
    fitness by -1 (if 1->0) or +1 (if 0->1).
    """

    @property
    def minimize(self) -> bool:
        return False

    def evaluate(self, solution: list[int]) -> float:
        return float(sum(solution))

    def delta_evaluate(self, solution: list[int], index: int) -> float | None:
        """O(1) delta evaluation for OneMax."""
        return -1.0 if solution[index] == 1 else 1.0
