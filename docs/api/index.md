---
description: Complete API reference for lonkit. Documentation for LON, CMLON, BasinHoppingSampler, LONVisualizer, and configuration classes.
---

# API Reference

Complete API documentation for lonkit.

## Modules

lonkit is organized into three main modules:

### [LON Module](lon.md)

Data structures for Local Optima Networks.

- [`LON`](lon.md#lonkit.lon.LON) - Local Optima Network representation
- [`CMLON`](lon.md#lonkit.lon.CMLON) - Compressed Monotonic LON
- [`LONConfig`](lon.md#lonkit.lon.LONConfig) - Configuration for LON construction

### [Sampling Module](sampling.md)

Basin-Hopping sampling for LON construction.

- [`compute_lon()`](sampling.md#lonkit.sampling.compute_lon) - High-level convenience function
- [`BasinHoppingSampler`](sampling.md#lonkit.sampling.BasinHoppingSampler) - Sampling class
- [`BasinHoppingSamplerConfig`](sampling.md#lonkit.sampling.BasinHoppingSamplerConfig) - Configuration

### [Visualization Module](visualization.md)

Plotting and animation tools.

- [`LONVisualizer`](visualization.md#lonkit.visualization.LONVisualizer) - Visualization class

### [Step Size Module](step_size.md)

Tools for estimating optimal Basin-Hopping step sizes.

- [`StepSizeEstimator`](step_size.md#lonkit.step_size.StepSizeEstimator) - Step size estimation
- [`StepSizeEstimatorConfig`](step_size.md#lonkit.step_size.StepSizeEstimatorConfig) - Configuration
- [`StepSizeResult`](step_size.md#lonkit.step_size.StepSizeResult) - Estimation result

## Quick Reference

### Creating a LON

```python
from lonkit import compute_lon, BasinHoppingSamplerConfig

# Simple usage
lon = compute_lon(
    func=objective_function,
    dim=2,
    lower_bound=-5.0,
    upper_bound=5.0,
    config=BasinHoppingSamplerConfig(n_runs=20, seed=42)
)
```

### Analyzing a LON

```python
# Basic properties
print(f"Optima: {lon.n_vertices}")
print(f"Edges: {lon.n_edges}")
print(f"Best: {lon.best_fitness}")

# Compute metrics
metrics = lon.compute_metrics()

# Convert to CMLON
cmlon = lon.to_cmlon()
cmlon_metrics = cmlon.compute_metrics()
```

### Visualizing a LON

```python
from lonkit import LONVisualizer

viz = LONVisualizer()

# 2D plot
viz.plot_2d(lon, output_path="lon.png")

# 3D plot
viz.plot_3d(lon, output_path="lon_3d.png")

# Animation
viz.create_rotation_gif(lon, output_path="lon.gif")

# All visualizations
viz.visualize_all(lon, output_folder="./output")
```
