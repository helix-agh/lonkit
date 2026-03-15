# API Reference

Complete API documentation for lonkit.

## Modules

lonkit is organized into the following modules:

### [LON Module](lon.md)

Data structures for Local Optima Networks.

- [`LON`](lon.md#lonkit.lon.LON) - Local Optima Network representation
- [`CMLON`](lon.md#lonkit.lon.CMLON) - Compressed Monotonic LON
- [`LONConfig`](lon.md#lonkit.lon.LONConfig) - Configuration for LON construction

### [Continuous Sampling Module](sampling.md)


- [`compute_lon()`](sampling.md#lonkit.continuous.sampling.compute_lon) - High-level convenience function
- [`BasinHoppingSampler`](sampling.md#lonkit.continuous.sampling.BasinHoppingSampler) - Sampling class
- [`BasinHoppingSamplerConfig`](sampling.md#lonkit.continuous.sampling.BasinHoppingSamplerConfig) - Configuration
- [`BasinHoppingResult`](sampling.md#lonkit.continuous.sampling.BasinHoppingResult) - Sampling result container
- [`StepSizeEstimator`](sampling.md#lonkit.continuous.step_size.StepSizeEstimator) - Step size estimation
- [`StepSizeEstimatorConfig`](sampling.md#lonkit.continuous.step_size.StepSizeEstimatorConfig) - Step size estimator configuration

### [Discrete Module](discrete.md)

Iterated Local Search sampling and built-in discrete problems.

- [`DiscreteProblem`](discrete.md#lonkit.discrete.problems.problem.DiscreteProblem) - Abstract base for discrete problems
- [`BitstringProblem`](discrete.md#lonkit.discrete.problems.bitstring.BitstringProblem) - Base class for bitstring problems
- [`NumberPartitioning`](discrete.md#lonkit.discrete.problems.bitstring.NumberPartitioning) - Number Partitioning Problem
- [`OneMax`](discrete.md#lonkit.discrete.problems.bitstring.OneMax) - OneMax benchmark
- [`ILSSampler`](discrete.md#lonkit.discrete.sampling.ILSSampler) - ILS sampling class
- [`ILSSamplerConfig`](discrete.md#lonkit.discrete.sampling.ILSSamplerConfig) - ILS configuration

### [Visualization Module](visualization.md)

Plotting and animation tools.

- [`LONVisualizer`](visualization.md#lonkit.visualization.LONVisualizer) - Visualization class

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
