import random

from lonkit import ILSSampler, ILSSamplerConfig, NumberPartitioning, OneMax


class TestOneMax4Exhaustive:
    """Exhaustive verification for OneMax(n=4) with 16 solutions."""

    def test_onemax_4_single_local_optimum(self):
        """OneMax(4) has exactly one local optimum: all-ones.

        From any solution, first-improvement HC with 1-bit-flip
        neighborhood always reaches [1,1,1,1] because every
        non-optimal solution has at least one 0-bit that can be
        flipped to improve fitness.
        """
        problem = OneMax(n=4)
        rng = random.Random(42)

        # Enumerate all 16 solutions
        all_solutions = [list(map(int, format(i, "04b"))) for i in range(16)]

        local_optima = set()
        for sol in all_solutions:
            opt, fit = problem.local_search(list(sol), rng)
            local_optima.add((problem.solution_id(opt), fit))

        assert len(local_optima) == 1
        assert local_optima == {("1111", 4.0)}

    def test_onemax_4_sampled_lon_single_node(self):
        """Sampled LON for OneMax(4) should have exactly 1 node.

        Since there's only one local optimum, all ILS transitions
        are self-loops (1111 -> 1111) which get removed during LON
        construction. The resulting LON has 1 node and 0 edges.
        """
        problem = OneMax(n=4)
        config = ILSSamplerConfig(n_runs=20, n_iter_no_change=10, seed=42)
        sampler = ILSSampler(config)
        result = sampler.sample(problem)
        lon = sampler.sample_to_lon(result)

        assert lon.n_vertices == 1
        assert lon.graph.ecount() == 0


class TestNPP6Exhaustive:
    """Exhaustive verification for NumberPartitioning(n=6) with 64 solutions."""

    def test_npp_6_exhaustive_local_optima(self):
        """Verify local optima count for NPP(n=6, k=0.5, instance_seed=1).

        Enumerate all 64 solutions, run local search from each, and verify
        that every discovered optimum is truly a local optimum (no improving
        single-bit-flip neighbor).
        """
        problem = NumberPartitioning(n=6, k=0.5, instance_seed=1)
        rng = random.Random(42)

        all_solutions = [list(map(int, format(i, "06b"))) for i in range(64)]

        local_optima = set()
        for sol in all_solutions:
            opt, _ = problem.local_search(list(sol), rng)
            local_optima.add(problem.solution_id(opt))

        assert len(local_optima) > 1, "NPP should have multiple local optima"

        # Verify each is truly a local optimum
        for sol_id in local_optima:
            sol = [int(b) for b in sol_id]
            fitness = problem.evaluate(sol)
            for i in range(problem.n):
                neighbor = list(sol)
                neighbor[i] = 1 - neighbor[i]
                neighbor_fitness = problem.evaluate(neighbor)
                assert not problem.is_better(neighbor_fitness, fitness), (
                    f"Solution {sol_id} is not a local optimum: "
                    f"flip at {i} gives fitness {neighbor_fitness} vs {fitness}"
                )

    def test_npp_6_sampled_lon_converges_to_exhaustive(self):
        """With enough runs, sampled LON should discover all local optima."""
        problem = NumberPartitioning(n=6, k=0.5, instance_seed=1)
        rng = random.Random(42)

        # Exhaustive: find all local optima
        all_solutions = [list(map(int, format(i, "06b"))) for i in range(64)]
        exhaustive_optima = set()
        for sol in all_solutions:
            opt, _ = problem.local_search(list(sol), rng)
            exhaustive_optima.add(problem.solution_id(opt))

        # Sampled: run many ILS runs
        config = ILSSamplerConfig(n_runs=200, n_iter_no_change=50, seed=42)
        sampler = ILSSampler(config)
        sample_problem = NumberPartitioning(n=6, k=0.5, instance_seed=1)
        result = sampler.sample(sample_problem)
        lon = sampler.sample_to_lon(result)

        sampled_optima = set(lon.graph.vs["name"])

        # Sampled should be a subset of exhaustive (can't discover non-optima)
        assert sampled_optima.issubset(exhaustive_optima)

        # With enough runs, we should discover most (if not all) optima
        coverage = len(sampled_optima) / len(exhaustive_optima)
        assert coverage == 1.0, (
            f"Only discovered {len(sampled_optima)}/{len(exhaustive_optima)} "
            f"optima ({coverage:.0%} coverage)"
        )
