"""Regression coverage for empty networks (issue #38)."""

import warnings

import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from lonkit import CMLON, LON, LONVisualizer

NETWORK_METRICS = {
    "n_optima": 0,
    "n_edges": 0,
    "n_funnels": 0,
    "n_global_funnels": 0,
    "neutral": 0.0,
    "global_strength": 0.0,
    "sink_strength": 0.0,
}
PERFORMANCE_METRICS = {"success": 0.0, "deviation": 0.0}


@pytest.fixture(params=["default", "filtered"])
def empty_lon(request) -> LON:
    if request.param == "default":
        return LON()
    graph = ig.Graph(n=1, directed=True)
    graph.vs["name"] = ["a"]
    graph.vs["Fitness"] = [1.0]
    graph.vs["Count"] = [1]
    graph.delete_vertices([0])
    return LON(graph=graph)


@pytest.mark.parametrize("known_best", [None, 0.0])
@pytest.mark.parametrize("run_values", [None, pd.Series(dtype=float)])
def test_empty_metrics(empty_lon, known_best, run_values):
    empty_lon.final_run_values = run_values
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert empty_lon.compute_network_metrics(known_best) == NETWORK_METRICS
        assert empty_lon.compute_performance_metrics(known_best) == PERFORMANCE_METRICS
        assert empty_lon.compute_metrics(known_best) == {**NETWORK_METRICS, **PERFORMANCE_METRICS}
        cmlon = empty_lon.to_cmlon()
        expected_network = {**NETWORK_METRICS, "global_funnel_proportion": 0.0}
        assert cmlon.compute_network_metrics(known_best) == expected_network
        assert cmlon.compute_performance_metrics(known_best) == PERFORMANCE_METRICS
        assert cmlon.compute_metrics(known_best) == {**expected_network, **PERFORMANCE_METRICS}


def test_empty_conversion_preserves_metadata(empty_lon):
    empty_lon.best_fitness = 3.0
    empty_lon.eq_atol = 1e-6
    empty_lon.minimize = False
    cmlon = empty_lon.to_cmlon()
    assert cmlon.n_vertices == cmlon.n_edges == 0
    assert cmlon.graph.is_directed()
    assert cmlon.graph is not empty_lon.graph
    assert cmlon.source_lon is empty_lon
    assert cmlon.best_fitness == 3.0
    assert cmlon.eq_atol == 1e-6
    assert cmlon.minimize is False


def test_standalone_empty_cmlon():
    cmlon = CMLON()
    assert cmlon.compute_metrics() == {**NETWORK_METRICS, "global_funnel_proportion": 0.0}
    assert cmlon.compute_performance_metrics() == {}
    assert cmlon.get_global_sinks() == cmlon.get_local_sinks() == []


def test_empty_vertex_properties(empty_lon):
    assert empty_lon.vertex_names == empty_lon.vertex_fitness == empty_lon.vertex_count == []
    cmlon = empty_lon.to_cmlon()
    assert cmlon.vertex_fitness == cmlon.vertex_count == []


@pytest.mark.parametrize("compressed", [False, True])
def test_empty_plots(empty_lon, compressed, tmp_path):
    network = empty_lon.to_cmlon() if compressed else empty_lon
    visualizer = LONVisualizer()
    assert visualizer.get_layout(network.graph).shape == (0, 2)
    output_path = tmp_path / "empty.png"
    fig = visualizer.plot_2d(network, output_path=output_path)
    try:
        assert output_path.stat().st_size > 0
        assert len(fig.axes[0].collections[0].get_offsets()) == 0
    finally:
        plt.close(fig)
    fig_3d = visualizer.plot_3d(network)
    assert len(fig_3d.data[0].x) == len(fig_3d.data[0].y) == len(fig_3d.data[0].z) == 0


def test_filtered_graph_retains_run_performance():
    lon = LON(best_fitness=1.0, final_run_values=pd.Series([1.0, 3.0]))
    for network in (lon, lon.to_cmlon()):
        assert network.compute_performance_metrics() == {"success": 0.5, "deviation": 1.0}


def test_empty_trace_has_descriptive_error():
    trace = pd.DataFrame(columns=["run", "fit1", "node1", "fit2", "node2"])
    with pytest.raises(ValueError, match="Trace is empty"):
        LON.from_trace_data(trace)
