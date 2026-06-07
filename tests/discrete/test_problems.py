import numpy as np
import pytest

from lonkit.discrete.problems.bitstring import (
    NKLandscape,
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


class TestNKLandscape:
    def test_minimize_is_false(self):
        p = NKLandscape(n=8, k=2, instance_seed=0)
        assert p.minimize is False

    def test_fitness_in_unit_interval(self):
        p = NKLandscape(n=10, k=3, instance_seed=1)
        rng = np.random.default_rng(7)
        for _ in range(50):
            sol = p.random_solution(rng)
            fit = p.evaluate(sol)
            assert 0.0 <= fit < 1.0

    def test_evaluate_is_deterministic(self):
        p = NKLandscape(n=8, k=2, instance_seed=3)
        sol = [0, 1, 1, 0, 1, 0, 0, 1]
        assert p.evaluate(sol) == p.evaluate(sol)

    def test_same_seed_same_instance(self):
        p1 = NKLandscape(n=8, k=2, instance_seed=42)
        p2 = NKLandscape(n=8, k=2, instance_seed=42)
        rng = np.random.default_rng(0)
        for _ in range(20):
            sol = p1.random_solution(rng)
            assert p1.evaluate(sol) == p2.evaluate(sol)

    def test_different_seed_different_instance(self):
        p1 = NKLandscape(n=8, k=2, instance_seed=1)
        p2 = NKLandscape(n=8, k=2, instance_seed=2)
        sol = [0, 1, 0, 1, 0, 1, 0, 1]
        assert p1.evaluate(sol) != p2.evaluate(sol)

    def test_adjacent_neighbors_structure(self):
        p = NKLandscape(n=5, k=2, instance_seed=0, neighbor_model="adjacent")
        assert p.neighbors[0] == [1, 2]
        assert p.neighbors[4] == [0, 1]  # cyclic wrap-around

    def test_random_neighbors_excludes_self_and_distinct(self):
        p = NKLandscape(n=8, k=3, instance_seed=5, neighbor_model="random")
        for i in range(8):
            assert i not in p.neighbors[i]
            assert len(set(p.neighbors[i])) == 3

    @pytest.mark.parametrize("neighbor_model", ["adjacent", "random"])
    def test_delta_evaluate_matches_full_evaluation(self, neighbor_model):
        p = NKLandscape(n=10, k=3, instance_seed=11, neighbor_model=neighbor_model)
        rng = np.random.default_rng(2)
        for _ in range(20):
            sol = p.random_solution(rng)
            base = p.evaluate(sol)
            for i in range(p.n):
                delta = p.delta_evaluate(sol, i)
                flipped = list(sol)
                flipped[i] = 1 - flipped[i]
                expected = p.evaluate(flipped) - base
                assert delta == pytest.approx(expected, abs=1e-12)

    def test_delta_evaluate_does_not_modify_solution(self):
        p = NKLandscape(n=6, k=2, instance_seed=4)
        sol = [1, 0, 1, 0, 1, 0]
        original = list(sol)
        p.delta_evaluate(sol, 3)
        assert sol == original

    def test_local_search_returns_local_optimum(self):
        p = NKLandscape(n=10, k=2, instance_seed=9)
        rng = np.random.default_rng(3)
        sol, fit = p.local_search(p.random_solution(rng), rng)
        # No single bit flip should improve a local optimum.
        for i in range(p.n):
            assert not p.is_better(fit + p.delta_evaluate(sol, i), fit)

    def test_k_zero_is_smooth(self):
        # With k=0 each bit is independent; greedy local search reaches the
        # global optimum regardless of starting point.
        p = NKLandscape(n=8, k=0, instance_seed=6)
        rng = np.random.default_rng(1)
        _, fit_a = p.local_search([0] * 8, rng)
        _, fit_b = p.local_search([1] * 8, rng)
        assert fit_a == pytest.approx(fit_b)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be in"):
            NKLandscape(n=4, k=4, instance_seed=0)
        with pytest.raises(ValueError, match="k must be in"):
            NKLandscape(n=4, k=-1, instance_seed=0)

    def test_instance_seed_is_required(self):
        with pytest.raises(ValueError, match="instance_seed is required"):
            NKLandscape(n=4, k=1, instance_seed=None)  # type: ignore[arg-type]

    def test_invalid_neighbor_model_raises(self):
        with pytest.raises(ValueError, match="neighbor_model"):
            NKLandscape(n=4, k=1, instance_seed=0, neighbor_model="circular")
