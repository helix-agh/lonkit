import numpy as np

from lonkit.discrete.problems.bitstring import (
    NumberPartitioning,
    OneMax,
)


class TestOneMax:
    def test_evaluate_all_ones(self):
        p = OneMax(n=4)
        assert p.evaluate([1, 1, 1, 1]) == 4.0

    def test_evaluate_all_zeros(self):
        p = OneMax(n=4)
        assert p.evaluate([0, 0, 0, 0]) == 0.0

    def test_minimize_is_false(self):
        p = OneMax(n=4)
        assert p.minimize is False

    def test_is_better_higher_is_better(self):
        p = OneMax(n=4)
        assert p.is_better(4, 3) is True
        assert p.is_better(3, 4) is False
        assert p.is_better(3, 3) is False

    def test_solution_id(self):
        p = OneMax(n=4)
        assert p.solution_id([0, 1, 0, 1]) == "0101"

    def test_reaches_global_optimum_from_zeros(self):
        p = OneMax(n=4)
        rng = np.random.default_rng(42)
        sol, fit = p.local_search([0, 0, 0, 0], rng)
        assert sol == [1, 1, 1, 1]
        assert fit == 4.0

    def test_perturb_does_not_modify_original(self):
        p = OneMax(n=4, n_perturbation_flips=2)
        original = [1, 1, 1, 1]
        copy = list(original)
        rng = np.random.default_rng(42)
        p.perturb(original, rng)
        assert original == copy

    def test_compare(self):
        p = OneMax(n=4)
        assert p.compare(4, 3) == 1
        assert p.compare(3, 4) == -1
        assert p.compare(3, 3) == 0


class TestNumberPartitioningBasic:
    def test_minimize_is_true(self):
        p = NumberPartitioning(n=4, weights=[3, 1, 1, 3])
        assert p.minimize is True

    def test_perfect_partition(self):
        p = NumberPartitioning(n=4, weights=[3, 1, 1, 3])
        # [0, 1, 0, 1] → A={3,1}=4, B={1,3}=4 → |4-4|=0 (perfect!)
        assert p.evaluate([0, 1, 0, 1]) == 0.0

    def test_is_better_lower_is_better(self):
        p = NumberPartitioning(n=4, weights=[3, 1, 1, 3])
        assert p.is_better(0, 5) is True
        assert p.is_better(5, 0) is False

    def test_compare_minimization(self):
        p = NumberPartitioning(n=4, weights=[1, 1, 1, 1])
        assert p.compare(0, 5) == 1
        assert p.compare(5, 0) == -1
        assert p.compare(3, 3) == 0
