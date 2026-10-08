from pathlib import Path

import matplotlib.pyplot as plt
from npp_paths import IMAGES_DIR
from PIL import Image, ImageOps

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

TITLE_FONTSIZE = 20
SUPTITLE_FONTSIZE = 26
AXIS_TITLE_FONTSIZE_3D = 26  # px
TICK_FONTSIZE_3D = 18


def render_3d_lons(cmlon_by_k: dict[float, CMLON], output_dir: Path = Path(IMAGES_DIR)) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.08),
        eye=dict(x=1.5, y=1.5, z=0.45),
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
        tickfont=dict(size=TICK_FONTSIZE_3D),
    )

    for k, cmlon in cmlon_by_k.items():
        vis = LONVisualizer(min_edge_width=0.5, max_edge_width=1, min_node_size=2.5, arrow_size=0.1)
        fig = vis.plot_3d(cmlon, seed=RANDOM_SEED)
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    **axis_config, title=dict(text="X", font=dict(size=AXIS_TITLE_FONTSIZE_3D))
                ),
                yaxis=dict(
                    **axis_config, title=dict(text="Y", font=dict(size=AXIS_TITLE_FONTSIZE_3D))
                ),
                zaxis=dict(
                    **axis_config,
                    nticks=6,
                    tickformat="~s",
                    title=dict(text="Fitness", font=dict(size=AXIS_TITLE_FONTSIZE_3D)),
                ),
                camera=dict(**standard_camera),
                aspectmode="cube",
            ),
            showlegend=False,
            width=850,
            height=850,
            margin=dict(l=10, r=10, t=10, b=10),
        )

        fig.write_image(output_dir / f"NPP_{k}_3d.png", scale=2)


def _load_trimmed(path: Path) -> Image.Image:
    """Load a rendered panel with its white border cropped, so the plot fills its grid cell."""
    img = Image.open(path).convert("RGB")
    return img.crop(ImageOps.invert(img).getbbox())


def render_merged_lon_grid(k_values: list[float], output_dir: Path = Path(IMAGES_DIR)) -> None:
    fig, axes = plt.subplots(2, len(k_values), figsize=(6 * len(k_values), 11))
    if len(k_values) == 1:
        axes = [[axes[0]], [axes[1]]]

    for idx, k in enumerate(k_values):
        img_2d = _load_trimmed(output_dir / f"NPP_{k}_2d.png")
        img_3d = _load_trimmed(output_dir / f"NPP_{k}_3d.png")

        ax_top = axes[0][idx]
        ax_top.imshow(img_2d)
        ax_top.set_title(f"2D CMLON (k={k})", fontsize=TITLE_FONTSIZE)
        ax_top.axis("off")

        ax_bottom = axes[1][idx]
        ax_bottom.imshow(img_3d)
        ax_bottom.set_title(f"3D CMLON (k={k})", fontsize=TITLE_FONTSIZE)
        ax_bottom.axis("off")

    fig.suptitle("Number Partitioning CMLON Views", fontsize=SUPTITLE_FONTSIZE, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=1.0)
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
        vis.plot_2d(cmlon, f"{IMAGES_DIR}/NPP_{k}_2d.png", dpi=250, seed=RANDOM_SEED)

    render_3d_lons(cmlon_by_k)
    render_merged_lon_grid(K_VALUES)


if __name__ == "__main__":
    main()
