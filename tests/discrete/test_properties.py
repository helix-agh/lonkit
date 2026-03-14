import pytest

from lonkit import ILSSampler, ILSSamplerConfig, NumberPartitioning


class TestLONProperties:
    """Properties that must hold for any LON built from ILS trace data."""

    @pytest.fixture(scope="class")
    def npp_result(self):
        """Run ILS sampling and return (problem, result, lon) tuple."""
        problem = NumberPartitioning(n=15, k=0.5, instance_seed=1)
        config = ILSSamplerConfig(n_runs=20, n_iter_no_change=50, seed=42)
        sampler = ILSSampler(config)
        result = sampler.sample(problem)
        lon = sampler.sample_to_lon(result)
        return problem, result, lon

    @pytest.fixture(scope="class")
    def npp_problem(self, npp_result):
        return npp_result[0]

    @pytest.fixture(scope="class")
    def npp_lon(self, npp_result):
        return npp_result[2]

    def test_all_nodes_are_local_optima(self, npp_lon, npp_problem):
        """
        Every node in the LON should be a local optimum.

        Verify by checking that no single-bit-flip neighbor improves fitness.
        """
        for name in npp_lon.graph.vs["name"]:
            sol = [int(b) for b in name]
            fitness = npp_problem.evaluate(sol)
            for i in range(npp_problem.n):
                neighbor = list(sol)
                neighbor[i] = 1 - neighbor[i]
                neighbor_fitness = npp_problem.evaluate(neighbor)
                assert not npp_problem.is_better(
                    neighbor_fitness, fitness
                ), f"Node {name} is not a local optimum: flip at {i} gives {neighbor_fitness} vs {fitness}"

    def test_best_fitness_is_minimum_vertex_fitness(self, npp_lon):
        """best_fitness equals the minimum vertex fitness."""
        assert npp_lon.best_fitness == min(npp_lon.vertex_fitness)

    def test_no_self_loops(self, npp_lon):
        """LON should have no self-loops (removed during construction)."""
        assert not any(npp_lon.graph.is_loop())

    def test_trace_only_contains_accepted_transitions(self, npp_result):
        """trace_df should only contain accepted transitions."""
        _, result, _ = npp_result
        accepted_in_raw = [r for r in result.raw_records if r["accepted"]]
        assert len(result.trace_df) == len(accepted_in_raw)

    def test_all_raw_records_have_required_fields(self, npp_result):
        """Each raw record must have all required fields."""
        _, result, _ = npp_result
        required_fields = {
            "run",
            "iteration",
            "current_id",
            "current_fitness",
            "new_id",
            "new_fitness",
            "accepted",
        }
        for rec in result.raw_records:
            assert required_fields.issubset(
                rec.keys()
            ), f"Missing fields: {required_fields - rec.keys()}"

    def test_n_funnels_equals_n_sinks(self, npp_lon):
        """n_funnels metric should equal the number of sinks."""
        metrics = npp_lon.compute_network_metrics()
        assert metrics["n_funnels"] == len(npp_lon.get_sinks())

    def test_n_global_funnels_leq_n_funnels(self, npp_lon):
        """Number of global funnels <= total number of funnels."""
        metrics = npp_lon.compute_network_metrics()
        assert metrics["n_global_funnels"] <= metrics["n_funnels"]


class TestCMLONProperties:
    """Properties that must hold for any CMLON."""

    @pytest.fixture(scope="class")
    def npp_lon_and_cmlon(self):
        """Build LON and CMLON from NPP sampling."""
        problem = NumberPartitioning(n=15, k=0.5, instance_seed=1)
        config = ILSSamplerConfig(n_runs=20, n_iter_no_change=50, seed=42)
        sampler = ILSSampler(config)
        result = sampler.sample(problem)
        lon = sampler.sample_to_lon(result)
        cmlon = lon.to_cmlon()
        return lon, cmlon

    @pytest.fixture(scope="class")
    def npp_lon(self, npp_lon_and_cmlon):
        return npp_lon_and_cmlon[0]

    @pytest.fixture(scope="class")
    def npp_cmlon(self, npp_lon_and_cmlon):
        return npp_lon_and_cmlon[1]

    def test_cmlon_has_no_equal_fitness_edges(self, npp_cmlon):
        """CMLON should have no edges between equal-fitness nodes."""
        el = npp_cmlon.graph.get_edgelist()
        fits = npp_cmlon.vertex_fitness
        for src, tgt in el:
            assert fits[src] != fits[tgt], (
                f"Equal-fitness edge found: "
                f"v{src} (fit={fits[src]}) -> v{tgt} (fit={fits[tgt]})"
            )

    def test_cmlon_fewer_or_equal_vertices(self, npp_lon, npp_cmlon):
        """CMLON has at most as many vertices as the source LON."""
        assert npp_cmlon.n_vertices <= npp_lon.n_vertices

    def test_cmlon_preserves_best_fitness(self, npp_lon, npp_cmlon):
        """CMLON best_fitness matches source LON best_fitness."""
        assert npp_cmlon.best_fitness == npp_lon.best_fitness

    def test_cmlon_no_self_loops(self, npp_cmlon):
        """CMLON should have no self-loops after contraction."""
        assert not any(npp_cmlon.graph.is_loop())
