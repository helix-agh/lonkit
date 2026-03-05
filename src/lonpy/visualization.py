import random
import shutil
from pathlib import Path

import igraph as ig
import imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from lonpy.lon import CMLON, LON

COLORS = {
    "global_optimum": "red",
    "local_sink": "royalblue",
    "global_basin": "pink",
    "local_basin": "lightskyblue",
    "edge": "dimgray",
    "lon_global": "red",
    "lon_local": "pink",
}

BACKGROUND_COLOR = "rgba(255,255,255,255)"


def _ensure_parent_dir(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


class LONVisualizer:
    """
    Visualizer for Local Optima Networks.

    Produces 2D and 3D visualizations of LON and CMLON graphs,
    including static images and animated GIFs.

    Example:
        >>> viz = LONVisualizer()
        >>> viz.plot_2d(lon, output_path="lon.png")
        >>> viz.plot_3d(cmlon, output_path="cmlon_3d.png")
    """

    def __init__(
        self,
        min_edge_width: float = 1.0,
        max_edge_width: float = 3.0,
        min_node_size: float = 2.0,
        max_node_size: float = 8.0,
        arrow_size: float = 0.2,
        alpha: int = 255,
    ):
        """
        Initialize visualizer.

        Args:
            min_edge_width: Minimum edge width. Default: `1.0`.
            max_edge_width: Maximum edge width. Default: `3.0`.
            min_node_size: Minimum node size. Default: `2.0`.
            max_node_size: Maximum node size. Default: `8.0`.
            arrow_size: Arrow size for directed edges. Default: `0.2`.
            alpha: Alpha value for colors (0-255). Default: `255`.
        """
        self.min_edge_width = min_edge_width
        self.max_edge_width = max_edge_width
        self.min_node_size = min_node_size
        self.max_node_size = max_node_size
        self.arrow_size = arrow_size
        self.alpha = alpha

    def compute_edge_widths(self, graph) -> list[float]:
        """Compute edge widths based on edge weight (Count attribute)."""
        if graph.ecount() == 0:
            return [1.0]

        counts = graph.es["Count"] if "Count" in graph.es.attributes() else [1] * graph.ecount()
        max_count = max(counts) if counts else 1

        widths = []
        for c in counts:
            w = (self.max_edge_width * c) / max_count
            w = max(w, self.min_edge_width)
            w = min(w, self.max_edge_width)
            widths.append(w)

        return widths

    def compute_node_sizes(self, graph) -> list[float]:
        """Compute node sizes based on incoming strength (weighted degree)."""
        if graph.ecount() == 0:
            return [self.max_node_size] * graph.vcount()

        weights = graph.es["Count"] if "Count" in graph.es.attributes() else None
        strengths = graph.strength(mode="in", weights=weights)

        sizes = []
        for s in strengths:
            size = 2 * s
            size = max(size, self.min_node_size)
            size = min(size, self.max_node_size)
            sizes.append(size)

        return sizes

    def compute_lon_colors(self, lon: LON) -> list[str]:
        """Compute node colors for LON visualization."""
        colors = []
        fits = lon.vertex_fitness
        best = lon.best_fitness

        for f in fits:
            if f == best:
                colors.append(COLORS["lon_global"])
            else:
                colors.append(COLORS["lon_local"])

        return colors

    def compute_cmlon_colors(self, cmlon: CMLON) -> list[str]:
        """Compute node colors for CMLON visualization."""
        n_vertices = cmlon.n_vertices
        colors = [COLORS["local_basin"]] * n_vertices

        sinks = cmlon.get_sinks()
        global_sinks = cmlon.get_global_sinks()
        fits = cmlon.vertex_fitness
        best = cmlon.best_fitness

        # Color global basins (nodes that can reach global optima)
        for gs in global_sinks:
            component = cmlon.graph.subcomponent(gs, mode="in")
            for v in component:
                colors[v] = COLORS["global_basin"]

        # Color suboptimal sinks
        for s in sinks:
            if fits[s] != best:
                colors[s] = COLORS["local_sink"]

        # Color global optima (overrides basin color)
        for i, f in enumerate(fits):
            if f == best:
                colors[i] = COLORS["global_optimum"]

        return colors

    def get_layout(self, graph, seed: int | None = None) -> np.ndarray:
        """Get 2D layout coordinates for graph nodes."""
        if seed is not None:
            np.random.seed(seed)
            ig.set_random_number_generator(random.Random(seed))
        layout = graph.layout_auto()
        return np.array(layout.coords)

    def plot_2d(
        self,
        lon_or_cmlon: LON | CMLON,
        output_path: str | Path | None = None,
        figsize: tuple[int, int] = (8, 8),
        dpi: int = 100,
        seed: int | None = None,
    ) -> plt.Figure:
        """
        Create 2D plot of LON or CMLON.

        Args:
            lon_or_cmlon: LON or CMLON instance.
            output_path: Path to save PNG. If `None`, only returns figure without saving. Default: `None`.
            figsize: Figure size in inches. Default: `(8, 8)`.
            dpi: DPI for output. Default: `100`.
            seed: Random seed for reproducible layout. Default: `None`.

        Returns:
            matplotlib Figure.
        """
        graph = lon_or_cmlon.graph

        edge_widths = self.compute_edge_widths(graph)
        node_sizes = self.compute_node_sizes(graph)

        node_colors = (
            self.compute_cmlon_colors(lon_or_cmlon)
            if isinstance(lon_or_cmlon, CMLON)
            else self.compute_lon_colors(lon_or_cmlon)
        )
        layout = self.get_layout(graph, seed=seed)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw edges
        if graph.ecount() > 0:
            for i, edge in enumerate(graph.es):
                src_idx = edge.source
                tgt_idx = edge.target

                x0, y0 = layout[src_idx]
                x1, y1 = layout[tgt_idx]

                # Draw arrow
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle=f"->,head_length={self.arrow_size},head_width={self.arrow_size}",
                        color=COLORS["edge"],
                        lw=edge_widths[i],
                        shrinkA=node_sizes[src_idx] * 2,
                        shrinkB=node_sizes[tgt_idx] * 2,
                    ),
                )

        # Scale node sizes for matplotlib
        scatter_sizes = [s**2 * 10 for s in node_sizes]

        ax.scatter(
            layout[:, 0],
            layout[:, 1],
            s=scatter_sizes,
            c=node_colors,
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        return fig

    def plot_3d(
        self,
        lon_or_cmlon: LON | CMLON,
        output_path: str | Path | None = None,
        width: int = 800,
        height: int = 800,
        seed: int | None = None,
    ) -> go.Figure:
        """
        Create 3D plot with fitness as Z-axis.

        Args:
            lon_or_cmlon: LON or CMLON instance.
            output_path: Path to save PNG. If `None`, only returns figure without saving. Default: `None`.
            width: Image width in pixels. Default: `800`.
            height: Image height in pixels. Default: `800`.
            seed: Random seed for reproducible layout. Default: `None`.

        Returns:
            plotly Figure.
        """
        graph = lon_or_cmlon.graph

        edge_widths = self.compute_edge_widths(graph)
        node_sizes = self.compute_node_sizes(graph)

        node_colors = (
            self.compute_cmlon_colors(lon_or_cmlon)
            if isinstance(lon_or_cmlon, CMLON)
            else self.compute_lon_colors(lon_or_cmlon)
        )

        layout = self.get_layout(graph, seed=seed)

        z_coords = np.array(lon_or_cmlon.vertex_fitness)

        x = layout[:, 0]
        y = layout[:, 1]
        z = z_coords

        fig = go.Figure()

        # Draw edges
        if graph.ecount() > 0:
            for i, edge in enumerate(graph.es):
                src_idx = edge.source
                tgt_idx = edge.target

                fig.add_trace(
                    go.Scatter3d(
                        x=[x[src_idx], x[tgt_idx]],
                        y=[y[src_idx], y[tgt_idx]],
                        z=[z[src_idx], z[tgt_idx]],
                        mode="lines",
                        line=dict(
                            color=COLORS["edge"],
                            width=edge_widths[i] * 2,
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

        # Draw nodes
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=np.array(node_sizes) * 2,
                    color=node_colors,
                    line=dict(color="black", width=0.5),
                ),
                hovertemplate="Fitness: %{z}<extra></extra>",
                showlegend=False,
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    title="",
                    showbackground=False,
                ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    title="",
                    showbackground=False,
                ),
                zaxis=dict(
                    showgrid=False,
                    title="",
                    showbackground=False,
                ),
                camera=dict(
                    eye=dict(x=0, y=-2, z=0.5),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.8),
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR,
        )

        if output_path:
            fig.write_image(str(output_path))

        return fig

    def create_rotation_gif(
        self,
        lon_or_cmlon: LON | CMLON,
        output_path: str | Path,
        duration: float = 3.0,
        fps: int = 10,
        width: int = 800,
        height: int = 800,
        seed: int | None = None,
        loop: int = 0,
        disposal: int = 2,
    ) -> None:
        """
        Create rotating GIF animation of 3D plot.

        Args:
            lon_or_cmlon: LON or CMLON instance.
            output_path: Path to save GIF.
            duration: Animation duration in seconds. Default: `3.0`.
            fps: Frames per second. Default: `10`.
            width: Image width in pixels. Default: `800`.
            height: Image height in pixels. Default: `800`.
            seed: Random seed for reproducible layout. Default: `None`.
            loop: GIF loop count (`0` = infinite). Default: `0`.
            disposal: GIF disposal method. Use `2` (restore to background) to avoid frame overlap. Default: `2`.
        """
        output_path = _ensure_parent_dir(output_path)
        graph = lon_or_cmlon.graph

        edge_widths = self.compute_edge_widths(graph)
        node_sizes = self.compute_node_sizes(graph)

        node_colors = (
            self.compute_cmlon_colors(lon_or_cmlon)
            if isinstance(lon_or_cmlon, CMLON)
            else self.compute_lon_colors(lon_or_cmlon)
        )

        layout = self.get_layout(graph, seed=seed)

        z_coords = np.array(lon_or_cmlon.vertex_fitness)
        x = layout[:, 0]
        y = layout[:, 1]
        z = z_coords

        total_frames = int(duration * fps)
        angles = np.linspace(0, 2 * np.pi, total_frames)

        frames: list[np.ndarray] = []

        try:
            for angle in angles:
                fig = go.Figure()

                # Draw edges
                if graph.ecount() > 0:
                    for j, edge in enumerate(graph.es):
                        src_idx = edge.source
                        tgt_idx = edge.target

                        fig.add_trace(
                            go.Scatter3d(
                                x=[x[src_idx], x[tgt_idx]],
                                y=[y[src_idx], y[tgt_idx]],
                                z=[z[src_idx], z[tgt_idx]],
                                mode="lines",
                                line=dict(color=COLORS["edge"], width=edge_widths[j] * 2),
                                hoverinfo="none",
                                showlegend=False,
                            )
                        )

                # Draw nodes
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(
                            size=np.array(node_sizes) * 2,
                            color=node_colors,
                            line=dict(color="black", width=0.5),
                        ),
                        showlegend=False,
                    )
                )

                # Rotating camera
                cam_dist = 2.0
                eye_x = cam_dist * np.sin(angle)
                eye_y = -cam_dist * np.cos(angle)
                eye_z = 0.5

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            title="",
                            showbackground=False,
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            title="",
                            showbackground=False,
                        ),
                        zaxis=dict(
                            showgrid=False,
                            title="",
                            showbackground=False,
                        ),
                        camera=dict(
                            eye=dict(x=eye_x, y=eye_y, z=eye_z),
                            up=dict(x=0, y=0, z=1),
                        ),
                        aspectmode="manual",
                        aspectratio=dict(x=1, y=1, z=0.8),
                    ),
                    width=width,
                    height=height,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor=BACKGROUND_COLOR,
                    plot_bgcolor=BACKGROUND_COLOR,
                )

                # Render directly to memory (avoids temp files and stray alpha blending)
                png_bytes = fig.to_image(format="png", width=width, height=height, scale=1)
                frames.append(imageio.v3.imread(png_bytes, extension=".png"))

            imageio.mimsave(
                str(output_path),
                frames,
                fps=fps,
                loop=loop,
                disposal=disposal,
            )

        finally:
            # Backwards-compat: older versions created temp frames on disk.
            temp_dir = Path(output_path).parent / ".temp_frames"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def visualize_all(
        self,
        lon: LON,
        output_folder: str | Path,
        seed: int | None = None,
    ) -> dict[str, Path]:
        """
        Create all visualizations for a LON.

        Generates a complete set of visualizations:
        - `lon.png`: 2D LON plot
        - `cmlon.png`: 2D CMLON plot
        - `3D_lon.png`: 3D LON plot
        - `3D_cmlon.png`: 3D CMLON plot
        - `lon.gif`: Rotating 3D LON animation
        - `cmlon.gif`: Rotating 3D CMLON animation

        Args:
            lon: LON instance.
            output_folder: Output directory path.
            seed: Random seed for reproducible layouts. Default: `None`.

        Returns:
            Dictionary mapping output type to file path.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Create CMLON
        cmlon = lon.to_cmlon()

        # 2D plots
        lon_2d_path = output_folder / "lon.png"
        self.plot_2d(lon, output_path=lon_2d_path, seed=seed)
        outputs["lon_2d"] = lon_2d_path
        plt.close()

        cmlon_2d_path = output_folder / "cmlon.png"
        self.plot_2d(cmlon, output_path=cmlon_2d_path, seed=seed)
        outputs["cmlon_2d"] = cmlon_2d_path
        plt.close()

        # 3D plots
        lon_3d_path = output_folder / "3D_lon.png"
        self.plot_3d(lon, output_path=lon_3d_path, seed=seed)
        outputs["lon_3d"] = lon_3d_path

        cmlon_3d_path = output_folder / "3D_cmlon.png"
        self.plot_3d(cmlon, output_path=cmlon_3d_path, seed=seed)
        outputs["cmlon_3d"] = cmlon_3d_path

        lon_gif_path = output_folder / "lon.gif"
        self.create_rotation_gif(lon, output_path=lon_gif_path, seed=seed)
        outputs["lon_gif"] = lon_gif_path

        cmlon_gif_path = output_folder / "cmlon.gif"
        self.create_rotation_gif(cmlon, output_path=cmlon_gif_path, seed=seed)
        outputs["cmlon_gif"] = cmlon_gif_path

        return outputs
