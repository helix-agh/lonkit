from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from problems import (
    ackley4,
    griewank,
    schwefel2_26,
)

IMAGES_DIR = "images"

FUNCTIONS = {
    "Ackley4": {
        "func": ackley4,
        "xmin": -35,
        "xmax": 35,
        "step": 0.8,
    },
    "Griewank": {
        "func": griewank,
        "xmin": -200,
        "xmax": 200,
        "step": 2,
    },
    "Schwefel 2.26": {"func": schwefel2_26, "xmin": -500, "xmax": 500, "step": 10},
}


def plot_all_surfaces(save: bool = True) -> None:
    fig = plt.figure(figsize=(16, 4))
    func_names = list(FUNCTIONS.keys())

    for idx, func_name in enumerate(func_names):
        cfg = FUNCTIONS[func_name]
        func = cfg["func"]
        xmin, xmax, step = cfg["xmin"], cfg["xmax"], cfg["step"]

        x = np.arange(xmin, xmax + step, step)
        y = np.arange(xmin, xmax + step, step)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="parula" if "parula" in plt.colormaps() else "viridis")

        ax.set_xlabel("$x_1$", fontsize=22)
        ax.set_ylabel("$x_2$", fontsize=22)
        ax.tick_params(labelsize=15)
        ax.grid(True, which="both")
        ax.view_init(elev=30, azim=55)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)

    plt.tight_layout()

    if save:
        Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{IMAGES_DIR}/fig1.png", dpi=150, bbox_inches="tight")
        print(f"Saved {IMAGES_DIR}/fig1.png")

    plt.show()


if __name__ == "__main__":
    plot_all_surfaces()
