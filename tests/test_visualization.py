import random
from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from lonkit import LON, LONVisualizer
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
