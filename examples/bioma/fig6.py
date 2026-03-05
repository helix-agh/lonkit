from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from problems import spread_spectrum_radar_polly_phase, ssc_ruspini
from utils import (
    IMAGES_DIR,
    FunctionConfig,
    build_all,
    save_metrics_figure,
    save_network_grid,
)

from lonpy import LONVisualizer

FUNCTIONS = {
    "SpreadSpectrumRadarPollyPhase": FunctionConfig(
        func=spread_spectrum_radar_polly_phase,
        bounds=(0, 2 * np.pi),
        step_size=1.2566,
        n_iter_no_change=100,
        coordinate_precision=1,
        dimensions=[3, 5, 8],
        best=0.5,
    ),
    "SSCRuspini": FunctionConfig(
        func=ssc_ruspini,
        bounds=(0, 200),
        step_size=48.64,
        n_iter_no_change=100,
        coordinate_precision=0,
        dimensions=[4, 6, 8],
        best=None,
    ),
}

FUNC_STYLES = {
    "SpreadSpectrumRadarPollyPhase": {"color": "tab:red", "marker": "o"},
    "SSCRuspini": {"color": "tab:orange", "marker": "s"},
}


def save_individual_figures(results: dict, images_dir: Path) -> None:
    """Save individual CMLON network plots."""
    viz = LONVisualizer()
    for (func_name, n_var), (cmlon, _metrics) in results.items():
        path = images_dir / f"fig6_{func_name}_dim{n_var}.png"
        fig = viz.plot_2d(cmlon, output_path=str(path))
        plt.close(fig)
        print(f"  Saved {path}")


def main() -> None:
    images_dir = Path(IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)

    results = build_all(FUNCTIONS)

    save_individual_figures(results, images_dir)
    save_network_grid(results, FUNCTIONS, images_dir / "fig6.png")
    save_metrics_figure(results, FUNCTIONS, FUNC_STYLES, images_dir / "fig5.png")


if __name__ == "__main__":
    main()
