"""
Number Partitioning Problem: Phase Transition Sweep with Empirical Boundary Detection
======================================================================================
This companion script sweeps the phase-transition parameter k across its full
range [0, 1] and plots LON / CMLON network metrics, ILS success, and the
transition-rate derivative.

Hard-region boundaries are detected *empirically* from the sweep data rather than
being hardcoded.  The detection strategy follows the finite-size scaling approach
standard in the statistical-mechanics literature on phase transitions:

  The transition centre k* is located at the steepest descent of the ILS success
  rate — i.e. the k value where the smoothed curve of -d(success)/dk is maximal.
  This mirrors how Gent & Walsh (1998) locate the critical constrainedness value
  experimentally.

  Metrics use ``compute_metrics()`` with default ``known_best=None`` (best fitness
  in the sampled network).

Note on the lonkit k parameter vs. the Mertens kappa
------------------------------------------------------
The canonical NPP phase transition in the physics literature (Mertens 1998,
Borgs et al. 2001) is parameterised by kappa = log2(M) / n, where integers are
drawn from [1, 2^M] and n is the count.  The critical point is kappa_c = 1.
lonkit's NumberPartitioning uses a different k: the ratio of the largest number
to the sum of all numbers.  These are related but not directly interchangeable,
so the empirical approach taken here is more appropriate than translating
kappa_c = 1 directly into lonkit's k space.

References
----------
Mertens, S. (1998). Phase transition in the number partitioning problem.
    Physical Review Letters, 81(20), 4281-4284.
Borgs, C., Chayes, J., Mertens, S., & Nair, C. (2001). Phase transition and
    finite-size scaling for the integer partitioning problem.
    Random Structures & Algorithms, 19(3-4), 261-294.
Gent, I. P., & Walsh, T. (1998). Analysis of heuristics for number partitioning.
    Computational Intelligence, 14(2), 430-451.

Outputs two figures: ``npp_phase_transition_sweep.png`` (single column) and
``npp_phase_transition_sweep_grid.png`` (two rows of three panels, legend below).

Requirements
------------
    pip install lonkit matplotlib numpy scipy
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from utils import IMAGES_DIR

from lonkit import ILSSampler, ILSSamplerConfig, LONConfig, NumberPartitioning

# Hard region detection

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


# Visualisatoin


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

    # Marj hard region in the figure
    _X_K_RIGHT = float(ks.max()) + 0.02

    _YLABEL_KW = {
        "rotation": 90,
        "ha": "center",
        "va": "center",
        "labelpad": 20,
        "fontsize": 8,
    }

    SUCCESS_YLABEL = "ILS success rate\nvs. best fitness in\nthe sampled network"

    PANELS = [
        ("Global to local\n funnel proportion", global_funnel_prop, C_GLOBAL, "^"),
        ("Number of CMLON\nlocal optima", n_optima, C_LINE, "o"),
        ("Number of CMLON\nfunnels", n_funnels, C_SINK, "s"),
        (SUCCESS_YLABEL, ils_success, C_SUCCESS, "o"),
        ("Global CMLON\nstrength", global_strength, C_GLOBAL, "D"),
    ]

    _N_ROWS = len(PANELS) + 1  # panels + derivative
    fig, axes = plt.subplots(
        _N_ROWS,
        1,
        figsize=(9, 12),
        sharex=True,
        gridspec_kw={
            "hspace": 0.24,
            "height_ratios": [1, 1, 1, 1, 1, 0.72],
        },
    )

    def _draw_background(ax):
        ax.axvspan(k_centre, _X_K_RIGHT, color=C_BAND, alpha=0.55, zorder=0)
        ax.axvline(k_centre, color=C_REF, linewidth=0.9, linestyle="--", zorder=1)

    def _plot_metric_ax(ax, ylabel, series, colour, marker):
        _draw_background(ax)
        ax.plot(
            ks,
            series,
            color=colour,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )
        ax.set_ylabel(ylabel, **_YLABEL_KW)
        ax.tick_params(axis="both", labelsize=9)
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
            linewidth=1.5,
            zorder=3,
            label=r"$-\,d(\mathrm{success})/dk$  (smoothed)",
        )
        ax.set_ylabel("Transition\nrate", **_YLABEL_KW)
        ax.set_xlabel("Phase-transition parameter  k", fontsize=10.5, labelpad=8)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            fontsize=7.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.38),
            ncol=1,
            framealpha=0.85,
            edgecolor="#d1d5db",
        )
        ax.set_xlim(ks.min() - 0.02, ks.max() + 0.02)

    # ── Panels in fixed order, then derivative ─

    for ax, panel in zip(axes[:-1], PANELS):
        _plot_metric_ax(ax, *panel)

    ax_d = axes[-1]
    _plot_derivative_ax(ax_d)

    # ── Legend ────────────────────────────────────────────────────────────────────

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
    ]
    axes[0].legend(
        handles=legend_handles,
        fontsize=7,
        loc="upper left",
        ncol=1,
        framealpha=0.9,
        edgecolor="#d1d5db",
        borderpad=0.35,
        labelspacing=0.35,
    )

    # ── Title and save ────────────────────────────────────────────────────────────

    fig.suptitle(
        f"NPP phase transition - LON metric analysis (N={N}, {N_RUNS} ILS runs per k)",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    # Room for suptitle, y-labels, and reference labels below the bottom axis (axes coords).
    fig.subplots_adjust(left=0.18, right=0.97, top=0.94, bottom=0.14, hspace=0.28)

    output_path = "npp_phase_transition_sweep.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"\nSaved \u2192 {output_path}")

    # ── Alternate layout: 2×3 panels + full-width legend row (two stacked entries) ─

    output_path_grid = Path(IMAGES_DIR / "npp_phase_transition_sweep_grid.png")
    fig_g = plt.figure(figsize=(14, 8))
    gs = fig_g.add_gridspec(
        3,
        3,
        hspace=0.38,
        wspace=0.36,
        height_ratios=[1, 1, 0.36],
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
        fontsize=10,
        loc="center",
        ncol=1,
        framealpha=0.95,
        edgecolor="#d1d5db",
        borderpad=0.45,
        labelspacing=0.65,
    )

    for ax in (ax_g00, ax_g01, ax_g02, ax_g10, ax_g11):
        ax.tick_params(axis="x", labelbottom=False)

    fig_g.suptitle(
        f"NPP phase transition - LON metric analysis (N={N}, {N_RUNS} ILS runs per k)",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig_g.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.11)
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


def main():
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Sweeping k across {len(K_VALUES)} values  (n={N}, n_runs={N_RUNS})\n")

    records = []
    sampler_config = ILSSamplerConfig(n_runs=N_RUNS, n_iter_no_change=N_ITER, seed=RANDOM_SEED)
    lon_config = LONConfig(eq_atol=EQ_ATOL)

    for k in K_VALUES:
        problem = NumberPartitioning(n=N, k=k, instance_seed=INSTANCE_SEED)
        sampler = ILSSampler(sampler_config)
        result = sampler.sample(problem)

        lon = sampler.sample_to_lon(result, lon_config)
        cmlon = lon.to_cmlon()

        m = cmlon.compute_metrics()

        records.append(
            {
                "k": k,
                "n_optima": m["n_optima"],
                "n_funnels": m["n_funnels"],
                "global_strength": m["global_strength"],
                "global_funnel_prop": m["global_funnel_proportion"],
                "ils_success": m["success"],
            }
        )
        print(
            f"  k={k:.3f}  optima={m['n_optima']:>3}  funnels={m['n_funnels']:>2}  "
            f"success={m['success']:.0%}"
        )

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
