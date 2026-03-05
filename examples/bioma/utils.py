from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

from lonpy import CMLON, BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

DEFAULT_N_RUNS = 100
DEFAULT_FITNESS_PRECISION = 2
DEFAULT_COORDINATE_PRECISION = 2
DEFAULT_SEED = 42
IMAGES_DIR = "images"


@dataclass
class FunctionConfig:
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    step_size: float
    n_iter_no_change: int
    coordinate_precision: int = DEFAULT_COORDINATE_PRECISION
    dimensions: list[int] = field(default_factory=lambda: [3, 5, 8])
    best: float | None = None


def build_cmlon(
    func_cfg: FunctionConfig,
    n_var: int,
    *,
    n_runs: int = DEFAULT_N_RUNS,
    fitness_precision: int = DEFAULT_FITNESS_PRECISION,
    seed: int = DEFAULT_SEED,
) -> CMLON:
    lb, ub = func_cfg.bounds
    domain = [(lb, ub)] * n_var

    config = BasinHoppingSamplerConfig(
        n_runs=n_runs,
        n_iter_no_change=func_cfg.n_iter_no_change,
        step_mode="fixed",
        step_size=func_cfg.step_size,
        fitness_precision=fitness_precision,
        coordinate_precision=func_cfg.coordinate_precision,
        bounded=True,
        seed=seed,
    )

    sampler = BasinHoppingSampler(config)
    lon = sampler.sample_to_lon(
        func_cfg.func,
        domain,
    )
    return lon.to_cmlon()


METRIC_PANELS = [
    ("n_optima", "Nodes"),
    ("n_funnels", "Funnels"),
    ("neutral", "Neutral"),
    ("sink_strength", "Strength"),
    ("success", "Success"),
    ("deviation", "Deviation"),
]


def _build_one(
    func_name: str,
    func_cfg: FunctionConfig,
    n_var: int,
) -> tuple[str, int, CMLON, dict]:
    """Build a single CMLON and compute its metrics (top-level for pickling)."""
    print(f"Sampling {func_name} n={n_var} ...")
    cmlon = build_cmlon(func_cfg, n_var)
    metrics = cmlon.compute_metrics(known_best=func_cfg.best)
    return func_name, n_var, cmlon, metrics


def build_all(
    functions: dict[str, FunctionConfig],
) -> dict[tuple[str, int], tuple[CMLON, dict]]:
    """Build all CMLONs and collect metrics (in parallel across functions/dimensions)."""
    tasks = [
        (func_name, func_cfg, n_var)
        for func_name, func_cfg in functions.items()
        for n_var in func_cfg.dimensions
    ]

    results: dict[tuple[str, int], tuple[CMLON, dict]] = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_build_one, *t) for t in tasks]
        for future in futures:
            func_name, n_var, cmlon, metrics = future.result()
            results[(func_name, n_var)] = (cmlon, metrics)
    return results


def save_network_grid(
    results: dict[tuple[str, int], tuple[CMLON, dict]],
    functions: dict[str, FunctionConfig],
    output_path: Path,
    labels: str = "abcdefghijklmnop",
) -> None:
    """Save a combined grid of CMLON network plots."""
    viz = LONVisualizer()
    func_names = list(functions.keys())
    all_dims = [functions[fn].dimensions for fn in func_names]

    n_rows = len(func_names)
    n_cols = max(len(d) for d in all_dims)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 8 * n_rows),
        dpi=150,
        squeeze=False,
    )

    label_idx = 0
    for row, func_name in enumerate(func_names):
        dims = functions[func_name].dimensions
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(dims):
                ax.axis("off")
                continue

            n_var = dims[col]
            cmlon, metrics = results[(func_name, n_var)]
            success = metrics["success"]

            ax.set_aspect("equal")
            ax.axis("off")

            graph = cmlon.graph
            edge_widths = viz.compute_edge_widths(graph)
            node_sizes = viz.compute_node_sizes(graph)
            node_colors = viz.compute_cmlon_colors(cmlon)
            layout = viz.get_layout(graph, seed=None)

            if graph.ecount() > 0:
                for i, edge in enumerate(graph.es):
                    src_idx = edge.source
                    tgt_idx = edge.target
                    x0, y0 = layout[src_idx]
                    x1, y1 = layout[tgt_idx]
                    ax.annotate(
                        "",
                        xy=(x1, y1),
                        xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle=f"->,head_length={viz.arrow_size},head_width={viz.arrow_size}",
                            color="dimgray",
                            lw=edge_widths[i],
                            shrinkA=node_sizes[src_idx] * 2,
                            shrinkB=node_sizes[tgt_idx] * 2,
                        ),
                    )

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

            label = labels[label_idx]
            ax.set_title(
                f"({label}) {func_name}, $n$ = {n_var}, success = {success:.2f}",
                fontsize=10,
                pad=6,
            )
            label_idx += 1

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_metrics_figure(
    results: dict[tuple[str, int], tuple[CMLON, dict]],
    functions: dict[str, FunctionConfig],
    func_styles: dict[str, dict],
    output_path: Path,
) -> None:
    """Create a 2x3 grid comparing metrics across dimensions for all functions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    func_names = list(functions.keys())

    for panel_idx, (metric_key, metric_label) in enumerate(METRIC_PANELS):
        row, col = divmod(panel_idx, 3)
        ax = axes[row, col]

        for func_name in func_names:
            dims = functions[func_name].dimensions
            style = func_styles[func_name]
            values = [results[(func_name, d)][1][metric_key] for d in dims]
            ax.plot(
                dims,
                values,
                color=style["color"],
                marker=style["marker"],
                label=func_name,
                linewidth=1.5,
                markersize=7,
            )

        ax.set_xlabel("Dimension")
        ax.set_ylabel(metric_label)
        ax.set_xticks(sorted({d for fn in func_names for d in functions[fn].dimensions}))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(func_names),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")
