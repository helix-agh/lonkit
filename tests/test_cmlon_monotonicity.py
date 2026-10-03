import warnings

import numpy as np
import pandas as pd
import pytest

from lonkit import LON, BasinHoppingSampler, BasinHoppingSamplerConfig, LONConfig


def _trace(rows: list[tuple[int, float, str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["run", "fit1", "node1", "fit2", "node2"])


def _worsening_edges(lon, minimize: bool = True) -> list[tuple[str, str]]:
    names = lon.graph.vs["name"]
    fits = lon.vertex_fitness
    result = []
    for src, tgt in lon.graph.get_edgelist():
        worse = fits[tgt] > fits[src] if minimize else fits[tgt] < fits[src]
        if worse and not lon._allclose(fits[src], fits[tgt]):
            result.append((names[src], names[tgt]))
    return result


def _schwefel(x: np.ndarray) -> float:
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


class TestCMLONMonotonicity:
    """CMLON must be monotonic regardless of the input LON."""

    def test_non_elitist_trace_worsening_edge_removed(self):
        """A user-supplied non-monotonic trace (5 -> 3 -> 4 -> 1) yields a monotonic CMLON."""
        trace = _trace([(1, 5.0, "a", 3.0, "b"), (1, 3.0, "b", 4.0, "c"), (1, 4.0, "c", 1.0, "d")])
        lon = LON.from_trace_data(trace)
        assert _worsening_edges(lon) == [("b", "c")]

        with pytest.warns(UserWarning, match="worsening edge"):
            cmlon = lon.to_cmlon()

        assert _worsening_edges(cmlon) == []
        names = cmlon.graph.vs["name"]
        sinks = {names[s] for s in cmlon.get_sinks()}
        assert sinks == {"b", "d"}
        assert {names[s] for s in cmlon.get_local_sinks()} == {"b"}
        assert {names[s] for s in cmlon.get_global_sinks()} == {"d"}

        metrics = cmlon.compute_network_metrics()
        assert metrics["n_funnels"] == 2
        assert metrics["n_global_funnels"] == 1
        assert metrics["sink_strength"] == pytest.approx(0.5)
        assert metrics["global_funnel_proportion"] == pytest.approx(0.5)

    def test_non_elitist_trace_maximization(self):
        """Worsening edges are also removed when maximizing."""
        trace = _trace([(1, 1.0, "a", 3.0, "b"), (1, 3.0, "b", 2.0, "c"), (1, 2.0, "c", 5.0, "d")])
        lon = LON.from_trace_data(trace, config=LONConfig(minimize=False))

        with pytest.warns(UserWarning, match="worsening edge"):
            cmlon = lon.to_cmlon()

        assert _worsening_edges(cmlon, minimize=False) == []
        names = cmlon.graph.vs["name"]
        assert {names[s] for s in cmlon.get_sinks()} == {"b", "d"}

    def test_deduplication_worsening_edge_removed(self):
        """Min-aggregation of a node recorded with noisy fitness creates a worsening edge.

        Node "y" is recorded as 2.0 and 2.0 - 1e-10; with "min" aggregation it gets
        2.0 - 1e-10, below its successor "w" (2.0 - 5e-11), although every recorded
        transition was improving.
        """
        trace = _trace(
            [
                (1, 3.0, "x", 2.0, "y"),
                (1, 2.0, "y", 2.0 - 5e-11, "w"),
                (2, 4.0, "v", 2.0 - 1e-10, "y"),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # multiple-fitness warning
            lon = LON.from_trace_data(trace)
        assert _worsening_edges(lon) == [("y", "w")]

        with pytest.warns(UserWarning, match="worsening edge"):
            cmlon = lon.to_cmlon()

        assert _worsening_edges(cmlon) == []
        names = cmlon.graph.vs["name"]
        assert "y" in {names[s] for s in cmlon.get_sinks()}

    def test_monotonic_lon_emits_no_warning(self):
        """No warning and no edges lost for an already monotonic LON."""
        trace = _trace([(1, 5.0, "a", 3.0, "b"), (1, 3.0, "b", 3.0, "c"), (1, 3.0, "c", 1.0, "d")])
        lon = LON.from_trace_data(trace)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cmlon = lon.to_cmlon()

        assert cmlon.n_vertices == 3
        assert cmlon.n_edges == 2

    @staticmethod
    def _neutral_chain_lon(minimize: bool, target: float) -> LON:
        """Chain a -> b -> c -> d with steps of 0.75 * eq_atol, so each step is equal
        within tolerance but a and d are not. d has edges to e (fitness `target`) and to f.
        After contraction, the component {a, b, c, d} is represented by a's fitness (0).
        """
        scale = (1 if minimize else -1) * 1e-12
        trace = _trace(
            [
                (1, 0.0, "a", 0.75 * scale, "b"),
                (1, 0.75 * scale, "b", 1.5 * scale, "c"),
                (1, 1.5 * scale, "c", 2.25 * scale, "d"),
                (1, 2.25 * scale, "d", target * scale, "e"),
                (2, 2.25 * scale, "d", -4.0 * scale, "f"),
            ]
        )
        lon = LON.from_trace_data(trace, LONConfig(minimize=minimize))
        assert _worsening_edges(lon, minimize=minimize) == []
        return lon

    @pytest.mark.parametrize("minimize", [True, False])
    @pytest.mark.parametrize("target", [0.5, 0.0, -0.5])
    def test_compression_merges_edges_that_become_equal(self, minimize, target):
        """An improving edge that becomes equal after contraction is merged, not removed."""
        lon = self._neutral_chain_lon(minimize, target)
        original_edges = lon.graph.get_edgelist()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cmlon = lon.to_cmlon()

        names = cmlon.graph.vs["name"]
        assert cmlon.n_vertices == 2
        assert cmlon.graph.vs["Count"] == [5, 1]
        assert [(names[s], names[t]) for s, t in cmlon.graph.get_edgelist()] == [("a", "f")]
        assert cmlon.graph.es["Count"] == [1]
        assert cmlon.compute_network_metrics()["n_funnels"] == 1
        assert lon.graph.get_edgelist() == original_edges

    @pytest.mark.parametrize("minimize", [True, False])
    def test_compression_removes_edges_that_become_worsening(self, minimize):
        """An improving edge that becomes worsening after contraction is removed."""
        lon = self._neutral_chain_lon(minimize, target=1.1)
        original_edges = lon.graph.get_edgelist()

        with pytest.warns(UserWarning, match="1 worsening edge.*after neutral-component"):
            cmlon = lon.to_cmlon()

        names = cmlon.graph.vs["name"]
        assert cmlon.n_vertices == 3
        assert [(names[s], names[t]) for s, t in cmlon.graph.get_edgelist()] == [("a", "f")]
        assert cmlon.graph.es["Count"] == [1]
        assert _worsening_edges(cmlon, minimize=minimize) == []
        assert lon.graph.get_edgelist() == original_edges

    @pytest.mark.parametrize(
        ("aggregation", "expected_worsening"),
        [
            ("min", [("y", "w")]),
            ("max", [("u", "y")]),
            ("mean", [("u", "y"), ("y", "w")]),
            ("first", [("u", "y")]),
        ],
    )
    def test_deduplication_worsening_edge_removed_for_each_aggregation(
        self, aggregation, expected_worsening
    ):
        """Every fitness aggregation strategy can produce worsening edges; CMLON removes them.

        Node "y" is recorded as 2.0 (run 1) and 2.0 - 1e-10 (run 2). All recorded
        transitions are improving, but aggregated "y" lies above "u" and/or below "w".
        """
        trace = _trace(
            [
                (1, 3.0, "x", 2.0, "y"),
                (1, 2.0, "y", 2.0 - 2e-11, "w"),
                (2, 2.0 - 8e-11, "u", 2.0 - 1e-10, "y"),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # multiple-fitness warning
            lon = LON.from_trace_data(trace, LONConfig(fitness_aggregation=aggregation))
        assert sorted(_worsening_edges(lon)) == expected_worsening

        with pytest.warns(UserWarning, match=f"Removed {len(expected_worsening)} worsening edge"):
            cmlon = lon.to_cmlon()

        assert _worsening_edges(cmlon) == []

    @pytest.mark.parametrize("minimize", [True, False])
    def test_random_traces_satisfy_cmlon_invariants(self, minimize):
        """Random non-elitist traces with near-equal fitness values always give a valid CMLON."""
        rng = np.random.default_rng(12345)
        atol = 1e-12
        for _ in range(300):
            n_nodes = int(rng.integers(2, 12))
            # Few distinct levels plus sub-tolerance noise: produces equal edges,
            # non-transitive chains and fitness duplicates of the same node.
            levels = rng.integers(0, 4, size=n_nodes).astype(float)
            noise_steps = np.array([-1.5, -0.75, 0.0, 0.75, 1.5]) * atol
            rows = []
            for run in range(int(rng.integers(1, 4))):
                for _ in range(int(rng.integers(1, 8))):
                    i, j = rng.choice(n_nodes, size=2, replace=False)
                    fit_i = levels[i] + rng.choice(noise_steps)
                    fit_j = levels[j] + rng.choice(noise_steps)
                    rows.append((run, fit_i, f"n{i}", fit_j, f"n{j}"))
            trace = _trace(rows)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lon = LON.from_trace_data(trace, LONConfig(minimize=minimize, eq_atol=atol))
                original_edges = lon.graph.get_edgelist()
                original_fits = lon.vertex_fitness
                cmlon = lon.to_cmlon()

            fits = cmlon.vertex_fitness
            for src, tgt in cmlon.graph.get_edgelist():
                improves = fits[tgt] < fits[src] if minimize else fits[tgt] > fits[src]
                assert improves, trace
                assert not cmlon._allclose(fits[src], fits[tgt]), trace
            assert cmlon.graph.is_dag(), trace
            assert not any(cmlon.graph.is_loop()), trace
            assert sum(cmlon.graph.vs["Count"]) == lon.n_vertices, trace
            assert lon.graph.get_edgelist() == original_edges
            assert lon.vertex_fitness == original_fits

    def test_basin_hopping_default_precision_integration(self):
        """Integration: Schwefel 2.26 (D=3) with default fitness_precision=None.

        Whether numerical noise produces worsening input edges depends on the
        optimizer/platform. The deterministic deduplication test above is the
        regression test; here we check the full sampling-to-CMLON pipeline.
        """
        config = BasinHoppingSamplerConfig(n_runs=30, coordinate_precision=2, seed=2)
        sampler = BasinHoppingSampler(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lon = sampler.sample_to_lon(sampler.sample(_schwefel, [(-500.0, 500.0)] * 3))

        if _worsening_edges(lon):
            with pytest.warns(UserWarning, match="worsening edge"):
                cmlon = lon.to_cmlon()
        else:
            cmlon = lon.to_cmlon()

        assert cmlon.n_vertices > 0
        fits = cmlon.vertex_fitness
        for src, tgt in cmlon.graph.get_edgelist():
            assert fits[tgt] < fits[src]
            assert not cmlon._allclose(fits[src], fits[tgt])
