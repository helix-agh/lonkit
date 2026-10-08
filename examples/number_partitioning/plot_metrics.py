"""
Number Partitioning Problem: Phase Transition Sweep
===================================================
The example script sweeps the phase-transition parameter k across its full
range [0, 1] and plots LON / CMLON network metrics, ILS success, and the
transition-rate derivative.

The transition centre k* is located at the steepest descent of the ILS success
rate - i.e. the k value where the smoothed curve of -d(success)/dk is maximal.

References
----------
Mertens, S. (1998). Phase transition in the number partitioning problem.
    Physical Review Letters, 81(20), 4281-4284.
Borgs, C., Chayes, J., Mertens, S., & Nair, C. (2001). Phase transition and
    finite-size scaling for the integer partitioning problem.
    Random Structures & Algorithms, 19(3-4), 261-294.
Gent, I. P., & Walsh, T. (1998). Analysis of heuristics for number partitioning.
    Computational Intelligence, 14(2), 430-451.

Outputs one figure: ``NPP_phase_transition_sweep.png``
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from npp_paths import IMAGES_DIR
from scipy.ndimage import uniform_filter1d

from lonkit import ILSSampler, ILSSamplerConfig, LONConfig, NumberPartitioning

SMOOTH_WIDTH = 3  # uniform moving-average window (points); odd integer


def detect_via_derivative(ks, success, smooth_width=SMOOTH_WIDTH):
    """
    Locate k* from the ILS success curve: k at the maximum of smoothed -d(success)/dk
    (positive-clipped).  Returns the derivative series used for the bottom panel.
    """
    smoothed = uniform_filter1d(success.astype(float), size=smooth_width, mode="nearest")
    deriv = -np.gradient(smoothed, ks)
    deriv = np.clip(deriv, 0, None)

    peak_idx = int(np.argmax(deriv))
    k_centre = float(ks[peak_idx])

    return k_centre, deriv


def plot_npp_metrics(
    ks,
    n_optima,
    n_funnels,
    global_strength,
    global_funnel_prop,
    ils_success,
    k_centre,
    deriv_signal,
):
    C_LINE = "#1a1a2e"
    C_GLOBAL = "#2563eb"
    C_SINK = "#dc2626"
    C_SUCCESS = "#16a34a"
    C_DERIV = "#7c3aed"
    C_BAND = "#fef08a"
    C_REF = "#6b7280"

    # Mark the "hard" region in the figure

    _X_K_RIGHT = float(ks.max()) + 0.02

    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 13
    TICK_FONTSIZE = 12
    LEGEND_FONTSIZE = 13

    _YLABEL_KW = {
        "rotation": 90,
        "ha": "center",
        "va": "center",
        "labelpad": 10,
        "fontsize": LABEL_FONTSIZE,
    }
    _XLABEL = "k"

    # (title, y-axis label, series, colour, marker)
    PANELS = [
        ("(a) Global to local funnel proportion", "Proportion", global_funnel_prop, C_GLOBAL, "^"),
        ("(b) Number of CMLON local optima", "Local optima", n_optima, C_LINE, "o"),
        ("(c) Number of CMLON funnels", "Funnels", n_funnels, C_SINK, "s"),
        ("(d) ILS success vs. best sampled fitness", "Success rate", ils_success, C_SUCCESS, "o"),
        ("(e) Global CMLON strength", "Strength", global_strength, C_GLOBAL, "D"),
    ]

    def _draw_background(ax):
        ax.axvspan(k_centre, _X_K_RIGHT, color=C_BAND, alpha=0.55, zorder=0)
        ax.axvline(k_centre, color=C_REF, linewidth=0.9, linestyle="--", zorder=1)

    def _style_axes(ax, title, ylabel):
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
        ax.set_ylabel(ylabel, **_YLABEL_KW)
        ax.set_xlabel(_XLABEL, fontsize=LABEL_FONTSIZE, labelpad=4)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    def _plot_metric_ax(ax, title, ylabel, series, colour, marker):
        _draw_background(ax)
        ax.plot(
            ks,
            series,
            color=colour,
            marker=marker,
            markersize=5,
            linewidth=1.7,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )
        _style_axes(ax, title, ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        if series.min() >= 0:
            rng = series.max() - series.min()
            ax.set_ylim(bottom=-0.05 * rng if rng > 0 else -0.1)

    def _plot_derivative_ax(ax):
        _draw_background(ax)
        ax.plot(
            ks,
            deriv_signal,
            color=C_DERIV,
            linewidth=1.8,
            zorder=3,
        )
        _style_axes(ax, "(f) Transition rate", r"$-\,d(\mathrm{success})/dk$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(ks.min() - 0.02, ks.max() + 0.02)

    # Legend

    legend_handles = [
        mpatches.Patch(
            color=C_BAND,
            alpha=0.55,
            label=(f"hard region  k \u2208 [{k_centre:.2f},\u202f{ks.max():.2f}]  "),
        ),
        Line2D(
            [0],
            [0],
            color=C_REF,
            linestyle="--",
            linewidth=0.9,
            label=(f"transition centre  k*\u202f=\u202f{k_centre:.2f}  "),
        ),
        Line2D(
            [0],
            [0],
            color=C_DERIV,
            linewidth=1.8,
            label=r"$-\,d(\mathrm{success})/dk$  (smoothed, panel f)",
        ),
    ]

    # Grid arrengment

    output_path_grid = Path(IMAGES_DIR) / "NPP_phase_transition_sweep.png"
    fig_g = plt.figure(figsize=(15, 9.5))
    gs = fig_g.add_gridspec(
        3,
        3,
        hspace=0.5,
        wspace=0.32,
        height_ratios=[1, 1, 0.16],
    )

    ax_g00 = fig_g.add_subplot(gs[0, 0])
    _plot_metric_ax(ax_g00, *PANELS[0])
    ax_g01 = fig_g.add_subplot(gs[0, 1], sharex=ax_g00)
    _plot_metric_ax(ax_g01, *PANELS[1])
    ax_g02 = fig_g.add_subplot(gs[0, 2], sharex=ax_g00)
    _plot_metric_ax(ax_g02, *PANELS[2])

    ax_g10 = fig_g.add_subplot(gs[1, 0], sharex=ax_g00)
    _plot_metric_ax(ax_g10, *PANELS[3])
    ax_g11 = fig_g.add_subplot(gs[1, 1], sharex=ax_g00)
    _plot_metric_ax(ax_g11, *PANELS[4])
    ax_g12 = fig_g.add_subplot(gs[1, 2], sharex=ax_g00)
    _plot_derivative_ax(ax_g12)

    ax_g_leg = fig_g.add_subplot(gs[2, :])
    ax_g_leg.set_axis_off()
    ax_g_leg.legend(
        handles=legend_handles,
        fontsize=LEGEND_FONTSIZE,
        loc="center",
        ncol=len(legend_handles),
        framealpha=0.95,
        edgecolor="#d1d5db",
        borderpad=0.45,
        labelspacing=0.65,
    )

    # Titles and saving

    fig_g.suptitle(
        f"NPP phase transition - LON metric analysis (N={N}, {N_RUNS} ILS runs per k)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig_g.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.04)
    fig_g.savefig(output_path_grid, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig_g)
    print(f"Saved \u2192 {output_path_grid}")


N = 20
INSTANCE_SEED = 1
N_RUNS = 100
N_ITER = 500
RANDOM_SEED = 42
EQ_ATOL = 1e-8

K_VALUES = np.linspace(0.1, 1.0, 50)
# Cap on worker processes, so the sweep behaves on shared machines.
N_JOBS = int(os.environ.get("LONKIT_N_JOBS", "8"))


def sweep_one(k: float) -> dict:
    """Sample one NPP instance at the given k and return its CMLON metrics."""
    sampler_config = ILSSamplerConfig(n_runs=N_RUNS, n_iter_no_change=N_ITER, seed=RANDOM_SEED)
    lon_config = LONConfig(eq_atol=EQ_ATOL)

    problem = NumberPartitioning(n=N, k=k, instance_seed=INSTANCE_SEED)
    sampler = ILSSampler(sampler_config)
    result = sampler.sample(problem)

    lon = sampler.sample_to_lon(result, lon_config)
    cmlon = lon.to_cmlon()

    m = cmlon.compute_metrics()
    print(
        f"  k={k:.3f}  optima={m['n_optima']:>3}  funnels={m['n_funnels']:>2}  "
        f"success={m['success']:.0%}",
        flush=True,
    )
    return {
        "k": k,
        "n_optima": m["n_optima"],
        "n_funnels": m["n_funnels"],
        "global_strength": m["global_strength"],
        "global_funnel_prop": m["global_funnel_proportion"],
        "ils_success": m["success"],
    }


def main():
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    print(
        f"Sweeping k across {len(K_VALUES)} values  (n={N}, n_runs={N_RUNS}, "
        f"{N_JOBS} worker processes)\n"
    )

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        records = list(executor.map(sweep_one, K_VALUES))

    ks = np.array([r["k"] for r in records])
    n_optima = np.array([r["n_optima"] for r in records])
    n_funnels = np.array([r["n_funnels"] for r in records])
    global_strength = np.array([r["global_strength"] for r in records])
    global_funnel_prop = np.array([r["global_funnel_prop"] for r in records])
    ils_success = np.array([r["ils_success"] for r in records])

    k_centre, deriv_signal = detect_via_derivative(ks, ils_success)

    print("\nHard-region detection results")
    print(f"  Transition centre k*: {k_centre:.3f}")

    plot_npp_metrics(
        ks,
        n_optima,
        n_funnels,
        global_strength,
        global_funnel_prop,
        ils_success,
        k_centre,
        deriv_signal,
    )


if __name__ == "__main__":
    main()
