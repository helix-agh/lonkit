# Visualization Guide

This guide covers how to create visualizations of Local Optima Networks.

## Quick Start

```python
from lonpy import compute_lon, LONVisualizer, BasinHoppingSamplerConfig

# Build a LON
lon = compute_lon(my_func, dim=2, lower_bound=-5, upper_bound=5,
                 config=BasinHoppingSamplerConfig(n_runs=20))

# Create visualizer
viz = LONVisualizer()

# Generate plots
viz.plot_2d(lon, output_path="lon_2d.png")
viz.plot_3d(lon, output_path="lon_3d.png")
```

## 2D Network Plot

The 2D plot shows the network structure with automatic layout:

```python
fig = viz.plot_2d(
    lon,
    output_path="lon.png",
    figsize=(10, 10),  # Figure size in inches
    dpi=150,           # Resolution
    seed=42            # Reproducible layout
)
```

### Understanding the Plot

**For LON:**

- **Red nodes**: Global optima (best fitness)
- **Pink nodes**: Other local optima
- **Arrows**: Transitions between optima
- **Line width**: Transition frequency (thicker = more frequent)

**For CMLON:**

- **Red nodes**: Global optimum sinks
- **Blue nodes**: Local (suboptimal) sinks
- **Pink nodes**: In global funnel (can reach red)
- **Light blue nodes**: In local funnels (trapped)

## 3D Landscape Plot

The 3D plot uses fitness as the Z-axis:

```python
fig = viz.plot_3d(
    lon,
    output_path="lon_3d.png",
    width=1000,   # Pixels
    height=800,   # Pixels
    seed=42
)
```

### Interactive Viewing

Without `output_path`, you can interact with the plot:

```python
fig = viz.plot_3d(lon)
fig.show()  # Opens in browser
```

## Animated Rotation GIF

Create a rotating view of the 3D landscape:

```python
viz.create_rotation_gif(
    lon,
    output_path="lon_rotation.gif",
    duration=5.0,  # Seconds
    fps=15,        # Frames per second
    width=800,
    height=800,
    seed=42
)
```

### GIF Parameters

| Parameter  | Default | Description                                                           |
| ---------- | ------- | --------------------------------------------------------------------- |
| `duration` | 3.0     | Animation length in seconds                                           |
| `fps`      | 10      | Frames per second                                                     |
| `width`    | 800     | Image width in pixels                                                 |
| `height`   | 800     | Image height in pixels                                                |
| `seed`     | None    | Random seed for reproducible layout                                   |
| `loop`     | 0       | Loop count (0 = infinite)                                             |
| `disposal` | 2       | GIF disposal method (2 = restore to background, avoids frame overlap) |

## Generate All Visualizations

Create a complete set of visualizations:

```python
outputs = viz.visualize_all(
    lon,
    output_folder="./output",
    seed=42
)

print("Generated files:")
for name, path in outputs.items():
    print(f"  {name}: {path}")
```

This creates:

- `lon.png` - 2D LON plot
- `cmlon.png` - 2D CMLON plot
- `3D_lon.png` - 3D LON plot
- `3D_cmlon.png` - 3D CMLON plot
- `lon.gif` - Rotating LON animation
- `cmlon.gif` - Rotating CMLON animation

## Customizing Appearance

### Visualizer Settings

```python
viz = LONVisualizer(
    min_edge_width=0.5,    # Minimum edge line width
    max_edge_width=4.0,    # Maximum edge line width
    min_node_size=3.0,     # Minimum node size
    max_node_size=12.0,    # Maximum node size
    arrow_size=0.3,        # Arrow head size
)
```

### Node and Edge Sizing

Node sizes and edge widths are computed automatically based on:

- **Node size**: Incoming edge weight (strength)
- **Edge width**: Transition frequency (Count attribute)

```python
# Access computed values
edge_widths = viz.compute_edge_widths(lon.graph)
node_sizes = viz.compute_node_sizes(lon.graph)
```

## Comparing LON and CMLON

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# LON
viz.plot_2d(lon, seed=42)
plt.sca(axes[0])
axes[0].set_title(f"LON ({lon.n_vertices} nodes)")

# CMLON
cmlon = lon.to_cmlon()
viz.plot_2d(cmlon, seed=42)
plt.sca(axes[1])
axes[1].set_title(f"CMLON ({cmlon.n_vertices} nodes)")

plt.tight_layout()
plt.savefig("comparison.png")
```

## Color Scheme

The default color scheme:

```python
COLORS = {
    "global_optimum": "red",      # Best fitness sinks
    "local_sink": "royalblue",    # Suboptimal sinks
    "global_basin": "pink",       # Nodes in global funnel
    "local_basin": "lightskyblue", # Nodes in local funnels
    "edge": "dimgray",            # Edge color
    "lon_global": "red",          # LON global optima
    "lon_local": "pink",          # LON other optima
}
```

## Working with matplotlib

The 2D plot returns a matplotlib Figure:

```python
fig = viz.plot_2d(lon, seed=42)

# Add title
fig.axes[0].set_title("My LON Visualization")

# Adjust and save
fig.tight_layout()
fig.savefig("custom_lon.png", dpi=200)

# Don't forget to close
plt.close(fig)
```

## Working with Plotly

The 3D plot returns a plotly Figure:

```python
fig = viz.plot_3d(lon, seed=42)

# Customize camera
fig.update_layout(
    scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.0)
    )
)

# Save in different formats
fig.write_image("lon_3d.png")
fig.write_html("lon_3d.html")  # Interactive

# Show in notebook
fig.show()
```

## Reproducible Layouts

Use `seed` parameter for reproducible node positions:

```python
# Same layout every time
viz.plot_2d(lon, output_path="plot1.png", seed=42)
viz.plot_2d(lon, output_path="plot2.png", seed=42)  # Identical
```

## High-Quality Output

For publication-quality figures:

```python
# High-resolution 2D
viz.plot_2d(
    lon,
    output_path="publication.png",
    figsize=(12, 12),
    dpi=300
)

# High-resolution 3D
viz.plot_3d(
    lon,
    output_path="publication_3d.png",
    width=1600,
    height=1200
)
```

## Next Steps

- [Examples](examples.md) - Complete visualization workflows
- [API Reference](../api/visualization.md) - Full visualization API
