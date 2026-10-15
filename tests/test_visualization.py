import random
from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lonkit import LON, LONConfig, LONVisualizer
from lonkit.visualization import COLORS
from tests.conftest import SEED


@pytest.fixture
def visualizer() -> LONVisualizer:
    return LONVisualizer()


@pytest.mark.parametrize("extension", ["png", "gif"])
def test_3d_export(visualizer: LONVisualizer, tmp_path: Path, extension: str) -> None:
    graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=True)
    graph.vs["Fitness"] = [3.0, 1.0, 0.0]
    graph.vs["Count"] = [1, 1, 1]
    graph.es["Count"] = [1, 1]
    lon = LON(graph=graph, best_fitness=0.0)
    path = tmp_path / f"plot.{extension}"

    if extension == "png":
        visualizer.plot_3d(lon, output_path=path, width=200, height=200, seed=SEED)
    else:
        visualizer.create_rotation_gif(
            lon, path, width=200, height=200, duration=1, fps=3, seed=SEED
        )

    with Image.open(path) as image:
        assert image.format == extension.upper()
        assert image.size == (200, 200)
        assert image.n_frames == (1 if extension == "png" else 3)
        pixels = np.array(image.convert("RGB"))
        assert pixels.min() < pixels.max()
        if extension == "gif":
            image.seek(1)
            assert not np.array_equal(pixels, np.array(image.convert("RGB")))


class TestLayoutReproducibility:
    def test_layout_same_seed_gives_same_coordinates(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        coords1 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        coords2 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        np.testing.assert_array_equal(coords1, coords2)

    def test_layout_different_seeds_give_different_coordinates(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        coords1 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        coords2 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED + 1)
        assert not np.array_equal(coords1, coords2)

    def test_layout_does_not_change_global_numpy_rng(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        np.random.seed(123)
        expected = np.random.rand(5)

        np.random.seed(123)
        visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        np.testing.assert_array_equal(np.random.rand(5), expected)

    def test_layout_restores_igraph_rng(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        ig.set_random_number_generator(random)
        random.seed(123)
        expected = ig.Graph.Erdos_Renyi(n=50, m=100).get_edgelist()

        random.seed(123)
        visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        assert ig.Graph.Erdos_Renyi(n=50, m=100).get_edgelist() == expected

    def test_layout_same_seed_ignores_global_numpy_state(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        np.random.seed(1)
        coords1 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        np.random.seed(2)
        coords2 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        np.testing.assert_array_equal(coords1, coords2)


class TestPlot2DReproducibility:
    def test_plot_2d_same_seed_same_figure(
        self, sphere_lon: LON, visualizer: LONVisualizer
    ) -> None:
        fig1 = visualizer.plot_2d(sphere_lon, seed=SEED)
        fig1.canvas.draw()
        pixels1 = np.array(fig1.canvas.buffer_rgba())
        plt.close(fig1)

        fig2 = visualizer.plot_2d(sphere_lon, seed=SEED)
        fig2.canvas.draw()
        pixels2 = np.array(fig2.canvas.buffer_rgba())
        plt.close(fig2)

        np.testing.assert_array_equal(pixels1, pixels2)

    def test_plot_2d_saves_to_file(
        self, sphere_lon: LON, visualizer: LONVisualizer, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "plot.png"
        visualizer.plot_2d(sphere_lon, output_path=output_path, seed=SEED)
        plt.close("all")

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestColorsRespectTolerance:
    """Colors classify global optima with `eq_atol`, consistently with the metrics."""

    @staticmethod
    def _two_sink_lon() -> LON:
        trace = pd.DataFrame(
            {
                "run": [1, 2],
                "fit1": [5.0, 5.0],
                "node1": ["x", "y"],
                "fit2": [0.0, 0.05],
                "node2": ["a", "b"],
            }
        )
        return LON.from_trace_data(trace, LONConfig(eq_atol=0.1))

    def test_cmlon_sinks_within_tolerance_are_global(self, visualizer: LONVisualizer) -> None:
        cmlon = self._two_sink_lon().to_cmlon()
        sinks = cmlon.get_sinks()
        colors = visualizer.compute_cmlon_colors(cmlon)

        assert sorted(cmlon.get_global_sinks()) == sorted(sinks)
        assert cmlon.get_local_sinks() == []
        assert [colors[s] for s in sinks] == [COLORS["global_optimum"]] * len(sinks)
        assert cmlon.compute_network_metrics()["n_global_funnels"] == len(sinks)

    def test_lon_nodes_within_tolerance_are_global(self, visualizer: LONVisualizer) -> None:
        lon = self._two_sink_lon()
        colors = dict(zip(lon.vertex_names, visualizer.compute_lon_colors(lon)))

        assert colors["a"] == colors["b"] == COLORS["lon_global"]
        assert colors["x"] == colors["y"] == COLORS["lon_local"]


class TestLegend:
    @pytest.fixture
    def two_funnel_lon(self) -> LON:
        trace = pd.DataFrame(
            {
                "run": [0, 1],
                "fit1": [3.0, 4.0],
                "node1": ["a", "c"],
                "fit2": [0.0, 1.0],
                "node2": ["b", "d"],
            }
        )
        return LON.from_trace_data(trace)

    def test_2d_legend_lists_present_categories(
        self, two_funnel_lon: LON, visualizer: LONVisualizer
    ) -> None:
        lon_fig = visualizer.plot_2d(two_funnel_lon, seed=SEED)
        cmlon_fig = visualizer.plot_2d(two_funnel_lon.to_cmlon(), seed=SEED)
        try:
            lon_labels = [t.get_text() for t in lon_fig.axes[0].get_legend().get_texts()]
            cmlon_labels = [t.get_text() for t in cmlon_fig.axes[0].get_legend().get_texts()]
        finally:
            plt.close("all")

        assert lon_labels == ["Global optimum", "Local optimum"]
        assert cmlon_labels == ["Global optimum", "Global funnel", "Local sink", "Local funnel"]

    def test_3d_legend_lists_present_categories(
        self, two_funnel_lon: LON, visualizer: LONVisualizer
    ) -> None:
        single_trace = pd.DataFrame(
            {"run": [0], "fit1": [1.0], "node1": ["a"], "fit2": [0.0], "node2": ["b"]}
        )
        single_funnel = LON.from_trace_data(single_trace).to_cmlon()

        fig = visualizer.plot_3d(two_funnel_lon.to_cmlon(), seed=SEED)
        single_fig = visualizer.plot_3d(single_funnel, seed=SEED)

        assert [t.name for t in fig.data if t.showlegend is not False] == [
            "Global optimum",
            "Global funnel",
            "Local sink",
            "Local funnel",
        ]
        assert [t.name for t in single_fig.data if t.showlegend is not False] == [
            "Global optimum",
            "Global funnel",
        ]
