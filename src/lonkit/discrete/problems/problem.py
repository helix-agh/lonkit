from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

SolutionT = TypeVar("SolutionT")


class DiscreteProblem(ABC, Generic[SolutionT]):
    """
    Abstract base class for discrete optimization problems.

    A DiscreteProblem defines the search space, objective function,
    neighborhood structure, and perturbation operator. It is **stateless**:
    all randomness is injected via an `rng` parameter owned by the caller
    (typically the ILS sampler).

    Subclasses must implement all abstract methods to define:
    - Solution generation and identity
    - Fitness evaluation (pure function)
    - Local search (hill climbing with problem-specific neighborhood)
    - Perturbation (escape operator for ILS)

    The ILS sampler calls these methods without knowing the solution
    representation, neighborhood structure, or move operators.

    Type parameter:
        SolutionT: The solution representation type (e.g., list[int] for
        bitstrings, list[int] for permutations).
    """

    @property
    def minimize(self) -> bool:
        """Whether this is a minimization problem. Default: True."""
        return True

    def is_better(self, a: float, b: float) -> bool:
        """Return True if fitness `a` is strictly better than `b`."""
        return a < b if self.minimize else a > b

    def is_better_or_equal(self, a: float, b: float) -> bool:
        """Return True if fitness `a` is better than or equal to `b`."""
        return a <= b if self.minimize else a >= b

    def compare(self, a: float, b: float) -> int:
        """
        Compare two fitness values.

        Returns:
            1 if a is better, -1 if b is better, 0 if equal.
        """
        if self.is_better(a, b):
            return 1
        if self.is_better(b, a):
            return -1
        return 0

    @abstractmethod
    def random_solution(self, rng: np.random.Generator) -> SolutionT:
        """
        Generate a random initial solution.

        Args:
            rng: Random number generator (owned by the caller).

        Returns:
            A solution object (representation is problem-specific).
            The returned solution does NOT need to be a local optimum.
        """
        ...

    @abstractmethod
    def evaluate(self, solution: SolutionT) -> float:
        """
        Evaluate and return the fitness of a solution.

        This must be a **pure function**: same solution always returns
        the same fitness, no side effects.

        Args:
            solution: A solution object.

        Returns:
            Scalar fitness value.
        """
        ...

    @abstractmethod
    def local_search(
        self, solution: SolutionT, rng: np.random.Generator
    ) -> tuple[SolutionT, float]:
        """
        Run local search from a starting solution to find a local optimum.

        This encapsulates the full hill-climbing loop:
        neighborhood definition, scanning strategy (first/best improvement),
        and termination.

        **Stochastic local search note**: When using first-improvement with
        randomized scan order, the local optimum reached from a given
        starting solution depends on the RNG state. This means basin
        identity is sampler-state dependent -- the same starting solution
        may reach different local optima across runs. This is an intentional
        and well-understood property of stochastic local search in LON
        construction.

        Args:
            solution: Starting solution (not necessarily a local optimum).
            rng: Random number generator (owned by the caller).

        Returns:
            Tuple of (local_optimum_solution, fitness).
        """
        ...

    @abstractmethod
    def perturb(self, solution: SolutionT, rng: np.random.Generator) -> SolutionT:
        """
        Apply a perturbation to escape the current basin of attraction.

        The perturbation should be strong enough to potentially reach a
        different basin but not so strong as to be random restart.

        Args:
            solution: A local optimum solution.
            rng: Random number generator (owned by the caller).

        Returns:
            A perturbed solution (NOT necessarily a local optimum).
        """
        ...

    @abstractmethod
    def solution_id(self, solution: SolutionT) -> str:
        """
        Return a unique string identifier for this solution.

        Two solutions that represent the same point in the search space
        MUST return the same ID. Two different solutions MUST return
        different IDs.

        Args:
            solution: A solution object.

        Returns:
            String identifier (used as node ID in the LON).
        """
        ...
