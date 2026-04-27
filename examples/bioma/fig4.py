from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
from problems import ackley4, griewank, schwefel2_26
from utils import IMAGES_DIR, FunctionConfig, build_cmlon

from lonkit import LONVisualizer

FUNCTIONS = {
    "Ackley 4": FunctionConfig(
        func=ackley4,
        bounds=(-35, 35),
        step_size=1.631,
        n_iter_no_change=300,
    ),
    "Griewank": FunctionConfig(
        func=griewank,
        bounds=(-600, 600),
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

N_VAR = 5


def render_3d_cmlons(func_names, cmlons):
    standard_camera = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.55, y=1.55, z=0.4)
    )

    axis_config = dict(
        visible=True,
        showgrid=True,
        gridcolor="lightgray",
        showline=True,
        linecolor="black",
        showbackground=True,
        backgroundcolor="rgb(250, 250, 250)",
        zeroline=True,
        zerolinecolor="gray",
        showticklabels=True,
    )

    plot_paths = []

    for func_name in func_names:
        viz = LONVisualizer()
        fig = viz.plot_3d(cmlons[func_name])

        fig.update_layout(
            scene=dict(
                xaxis=dict(**axis_config, title="X"),
                yaxis=dict(**axis_config, title="Y"),
                zaxis=dict(**axis_config, title="Fitness"),
                camera=dict(**standard_camera),
                aspectmode="cube",
            ),
            title=dict(
                text=f"{func_name}",
                x=0.5,
                y=0.95,
                xanchor="center",
                font=dict(size=20),
            ),
            showlegend=False,
            width=600,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgb(255, 255, 255)",
            plot_bgcolor="rgb(255, 255, 255)",
        )

        path = Path(IMAGES_DIR) / f"fig4_{func_name.replace(' ', '_')}.png"
        fig.write_image(path, scale=2)
        plot_paths.append((func_name, path))
        print(f"Saved 3D panel to {path}")

    return plot_paths


def merge_3d_plots(plot_paths):
    merged_fig, axes = plt.subplots(1, len(plot_paths), figsize=(6 * len(plot_paths), 6))
    if len(plot_paths) == 1:
        axes = [axes]

    for ax, (_, image_path) in zip(axes, plot_paths):
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.axis("off")

    merged_fig.tight_layout()

    final_path = Path(IMAGES_DIR) / "fig4.png"
    merged_fig.savefig(final_path, dpi=200, bbox_inches="tight")
    plt.close(merged_fig)
    print(f"Successfully saved merged figure to {final_path}")


def main() -> None:
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    func_names = list(FUNCTIONS.keys())

    with ProcessPoolExecutor() as executor:
        futures = {
            func_name: executor.submit(build_cmlon, FUNCTIONS[func_name], N_VAR)
            for func_name in func_names
        }

        cmlons = {name: fut.result() for name, fut in futures.items()}

    plot_paths = render_3d_cmlons(func_names, cmlons)

    merge_3d_plots(plot_paths)


if __name__ == "__main__":
    main()
