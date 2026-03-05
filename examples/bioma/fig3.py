from pathlib import Path

import matplotlib.pyplot as plt
from problems import ackley4, griewank, schwefel2_26
from utils import (
    IMAGES_DIR,
    FunctionConfig,
    build_all,
    save_metrics_figure,
    save_network_grid,
)

from lonpy import LONVisualizer

FUNCTIONS = {
    "Ackley 4": FunctionConfig(
        func=ackley4,
        bounds=(-35, 35),
        step_size=1.631,
        n_iter_no_change=300,
    ),
    "Griewank": FunctionConfig(
        func=griewank,
        bounds=(-200, 200),
        step_size=3.6,
        n_iter_no_change=200,
    ),
    "Schwefel 2.26": FunctionConfig(
        func=schwefel2_26,
        bounds=(-500, 500),
        step_size=151.0,
        n_iter_no_change=4000,
    ),
}

FUNC_STYLES = {
    "Ackley 4": {"color": "tab:blue", "marker": "o"},
    "Griewank": {"color": "tab:purple", "marker": "P"},
    "Schwefel 2.26": {"color": "tab:green", "marker": "^"},
}


def save_individual_figures(results: dict, images_dir: Path) -> None:
    """Save individual CMLON network plots."""
    viz = LONVisualizer()
    for (func_name, n_var), (cmlon, _metrics) in results.items():
        path = images_dir / f"fig3_{func_name}_dim{n_var}.png"
        fig = viz.plot_2d(cmlon, output_path=str(path))
        plt.close(fig)
        print(f"  Saved {path}")


def main() -> None:
    images_dir = Path(IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)

    results = build_all(FUNCTIONS)

    save_individual_figures(results, images_dir)
    save_network_grid(results, FUNCTIONS, images_dir / "fig3.png")
    save_metrics_figure(results, FUNCTIONS, FUNC_STYLES, images_dir / "fig2.png")


if __name__ == "__main__":
    main()
