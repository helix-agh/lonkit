import math
import random as stdlib_random
import warnings

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

    def random_solution(self, rng: stdlib_random.Random) -> list[int]:
        return [rng.randint(0, 1) for _ in range(self.n)]

    def solution_id(self, solution: list[int]) -> str:
        return "".join(str(b) for b in solution)

    def perturb(self, solution: list[int], rng: stdlib_random.Random) -> list[int]:
        """Flip `n_perturbation_flips` random distinct bits."""
        sol = list(solution)  # copy
        indices = rng.sample(range(self.n), self.n_perturbation_flips)
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
        self, solution: list[int], rng: stdlib_random.Random
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
        first_improvement: Hill climbing strategy (default: True).
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
            # Generate item weights using a separate RNG (matches reference exactly)
            rng = stdlib_random.Random(instance_seed)
            upper = round(math.pow(2, n * k))
            self.weights = [rng.randrange(1, upper + 1) for _ in range(n)]
        else:
            raise ValueError("Provide either `weights` or both `k` and `instance_seed`")

    def evaluate(self, solution: list[int]) -> float:
        cost_a = sum(self.weights[i] for i in range(self.n) if solution[i] == 0)
        cost_b = sum(self.weights[i] for i in range(self.n) if solution[i] == 1)
        return float(abs(cost_a - cost_b))


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
