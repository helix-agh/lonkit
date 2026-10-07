from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

from lonkit import CMLON, BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

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
) -> tuple[CMLON, pd.DataFrame]:
    """Build a CMLON and return it alongside the raw sampling trace.

    The trace DataFrame (columns ``[run, fit1, node1, fit2, node2]``) is the
    reproducible source: ``LON.from_trace_data(trace)`` rebuilds the LON and CMLON.
    """
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
    result = sampler.sample(func_cfg.func, domain)
    lon = sampler.sample_to_lon(result)
    return lon.to_cmlon(), result.trace_df


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
) -> tuple[str, int, CMLON, dict, pd.DataFrame]:
    """Build a single CMLON and compute its metrics (top-level for pickling)."""
    print(f"Sampling {func_name} n={n_var} ...")
    cmlon, trace_df = build_cmlon(func_cfg, n_var)
    metrics = cmlon.compute_metrics(known_best=func_cfg.best)
    return func_name, n_var, cmlon, metrics, trace_df


def build_all(
    functions: dict[str, FunctionConfig],
    *,
    data_dir: Path | None = None,
) -> dict[tuple[str, int], tuple[CMLON, dict]]:
    """Build all CMLONs and collect metrics (in parallel across functions/dimensions).

    If ``data_dir`` is given, the raw sampling trace of each network and a
    ``metrics.csv`` summary are written there for reviewers to reanalyze.
    """
    tasks = [
        (func_name, func_cfg, n_var)
        for func_name, func_cfg in functions.items()
        for n_var in func_cfg.dimensions
    ]

    results: dict[tuple[str, int], tuple[CMLON, dict]] = {}
    traces: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_build_one, *t) for t in tasks]
        for future in futures:
            func_name, n_var, cmlon, metrics, trace_df = future.result()
            results[(func_name, n_var)] = (cmlon, metrics)
            traces[f"{_slug(func_name)}_dim{n_var}"] = trace_df

    if data_dir is not None:
        save_traces(traces, data_dir)
        save_metrics_csv(results, data_dir / "metrics.csv")

    return results


_TRACE_README = """\
# Raw sampling traces

Each `*.csv` is the raw Basin-Hopping sampling trace for one (function, dimension),
with columns `[run, fit1, node1, fit2, node2]`. The trace is the reproducible
source for every figure and metric: rebuild the LON/CMLON and reproduce the results
with a few lines.

```python
import pandas as pd
from lonkit import LON, LONVisualizer

trace = pd.read_csv("Ackley_4_dim3.csv")   # pick any file in this folder
lon = LON.from_trace_data(trace)
cmlon = lon.to_cmlon()
print(cmlon.compute_metrics())             # matches metrics.csv
LONVisualizer().plot_2d(cmlon)
```

`metrics.csv` (when present) lists the computed metrics per (function, dimension).
"""


def _slug(name: str) -> str:
    """Filesystem-safe stem for a function name (e.g. ``Ackley 4`` -> ``Ackley_4``)."""
    return name.replace(" ", "_")


def save_traces(traces: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Write one CSV per raw sampling trace, plus a README explaining reuse."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for stem, trace_df in traces.items():
        path = data_dir / f"{stem}.csv"
        trace_df.to_csv(path, index=False)
        print(f"Saved {path}")
    (data_dir / "README.md").write_text(_TRACE_README, encoding="utf-8")


def save_metrics_csv(
    results: dict[tuple[str, int], tuple[CMLON, dict]],
    output_path: Path,
) -> None:
    """Write a metrics.csv summary with one row per (function, dimension)."""
    rows = [
        {"function": func_name, "dimension": n_var, **metrics}
        for (func_name, n_var), (_cmlon, metrics) in results.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def save_network_grid(
    results: dict[tuple[str, int], tuple[CMLON, dict]],
    functions: dict[str, FunctionConfig],
    output_path: Path,
    labels: str = "abcdefghijklmnop",
    layout_seed: int = DEFAULT_SEED,
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
            layout = viz.get_layout(graph, seed=layout_seed)

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

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")
