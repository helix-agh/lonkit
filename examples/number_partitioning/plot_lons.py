from pathlib import Path

import matplotlib.pyplot as plt
from npp_paths import IMAGES_DIR

from lonkit import (
    CMLON,
    ILSSampler,
    ILSSamplerConfig,
    LONConfig,
    LONVisualizer,
    NumberPartitioning,
)

N = 20
INSTANCE_SEED = 1
N_RUNS = 100
N_ITER = 500
RANDOM_SEED = 42

K_VALUES = [0.3, 0.7, 0.95]


def render_3d_lons(cmlon_by_k: dict[float, CMLON], output_dir: Path = Path(IMAGES_DIR)) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.55, y=1.55, z=0.4),
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

    for k, cmlon in cmlon_by_k.items():
        vis = LONVisualizer(min_edge_width=0.5, max_edge_width=1, min_node_size=2.5, arrow_size=0.1)
        fig = vis.plot_3d(cmlon)
        fig.update_layout(
            scene=dict(
                xaxis=dict(**axis_config, title="X"),
                yaxis=dict(**axis_config, title="Y"),
                zaxis=dict(**axis_config, title="Fitness"),
                camera=dict(**standard_camera),
                aspectmode="cube",
            ),
            showlegend=False,
            width=900,
            height=700,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        fig.write_image(output_dir / f"NPP_{k}_3d.png", scale=2)


def render_merged_lon_grid(k_values: list[float], output_dir: Path = Path(IMAGES_DIR)) -> None:
    fig, axes = plt.subplots(2, len(k_values), figsize=(5 * len(k_values), 10))
    if len(k_values) == 1:
        axes = [[axes[0]], [axes[1]]]

    for idx, k in enumerate(k_values):
        img_2d = plt.imread(output_dir / f"NPP_{k}_2d.png")
        img_3d = plt.imread(output_dir / f"NPP_{k}_3d.png")

        ax_top = axes[0][idx]
        ax_top.imshow(img_2d)
        ax_top.set_title(f"2D CMLON (k={k})", fontsize=12)
        ax_top.axis("off")

        ax_bottom = axes[1][idx]
        ax_bottom.imshow(img_3d)
        ax_bottom.set_title(f"3D CMLON (k={k})", fontsize=12)
        ax_bottom.axis("off")

    fig.suptitle("Number Partitioning CMLON Views", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "NPP_merged_cmlon_views.png", dpi=200)
    plt.close(fig)


def main():
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    sampler_config = ILSSamplerConfig(n_runs=N_RUNS, n_iter_no_change=N_ITER, seed=RANDOM_SEED)

    lon_config = LONConfig(eq_atol=1e-8)
    cmlon_by_k = {}

    for k in K_VALUES:
        problem = NumberPartitioning(n=N, k=k, instance_seed=INSTANCE_SEED)
        sampler = ILSSampler(sampler_config)
        result = sampler.sample(problem)

        lon = sampler.sample_to_lon(result, lon_config)
        cmlon = lon.to_cmlon()
        cmlon_by_k[k] = cmlon

        vis = LONVisualizer(0.5, 1, arrow_size=0.1)
        vis.plot_2d(cmlon, f"{IMAGES_DIR}/NPP_{k}_2d.png")

    render_3d_lons(cmlon_by_k)
    render_merged_lon_grid(K_VALUES)


if __name__ == "__main__":
    main()
