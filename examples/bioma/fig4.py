from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from plotly.subplots import make_subplots
from problems import ackley4, griewank, schwefel2_26
from utils import IMAGES_DIR, FunctionConfig, build_cmlon

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

N_VAR = 5


def main() -> None:
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    viz = LONVisualizer()
    func_names = list(FUNCTIONS.keys())

    # Build CMLONs in parallel, then create 3D figures
    with ProcessPoolExecutor() as executor:
        futures = {
            func_name: executor.submit(build_cmlon, FUNCTIONS[func_name], N_VAR)
            for func_name in func_names
        }
        cmlons = {name: fut.result() for name, fut in futures.items()}

    figures = []
    for func_name in func_names:
        fig = viz.plot_3d(cmlons[func_name])
        figures.append(fig)

    # Combine into a single 1x3 subplot figure
    combined = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"(a) {func_names[0]}",
            f"(b) {func_names[1]}",
            f"(c) {func_names[2]}",
        ],
        horizontal_spacing=0.02,
    )

    for idx, fig in enumerate(figures, start=1):
        scene_name = f"scene{idx}" if idx > 1 else "scene"
        for trace in fig.data:
            trace.scene = scene_name
            combined.add_trace(trace, row=1, col=idx)
        # Copy camera / axis settings from original figure
        if "scene" in fig.layout:
            combined.layout[scene_name].update(fig.layout.scene)

    combined.update_layout(
        showlegend=False,
        width=1800,
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    combined.write_image(f"{IMAGES_DIR}/fig4.png", scale=2)
    combined.write_html(f"{IMAGES_DIR}/fig4.html")


if __name__ == "__main__":
    main()
