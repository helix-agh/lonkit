# lonpy

**Local Optima Networks for Continuous Optimization**

![lonpy](assets/icon.png){ width="100%" }

lonpy is a Python library for constructing, analyzing, and visualizing Local Optima Networks (LONs) for continuous optimization problems.

## What are Local Optima Networks?

Local Optima Networks (LONs) are graph-based models that capture the global structure of fitness landscapes. They help researchers and practitioners understand:

- **Landscape topology**: How local optima are distributed and connected
- **Search difficulty**: Whether the landscape has a single funnel or multiple competing basins
- **Algorithm behavior**: How optimization algorithms navigate between local optima

## Key Features

<div class="grid cards" markdown>

- **Basin-Hopping Sampling**

    ---

    Efficient exploration of fitness landscapes using configurable Basin-Hopping with customizable perturbation strategies

- **LON Construction**

    ---

    Automatic construction of Local Optima Networks from sampling data with support for both LON and CMLON representations

- **Rich Metrics**

    ---

    Compute landscape metrics including funnel analysis, neutrality measures, and strength metrics (global and sink)

- **Beautiful Visualizations**

    ---

    2D and 3D plots with support for animated GIFs showing the landscape structure

</div>

## Quick Example

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer, BasinHoppingSamplerConfig

# Define the Rastrigin function
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Build the LON
config = BasinHoppingSamplerConfig(n_runs=20, seed=42)
lon = compute_lon(
    rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    config=config
)

# Analyze
metrics = lon.compute_metrics()
print(f"Found {lon.n_vertices} local optima")
print(f"Landscape has {metrics['n_funnels']} funnels")

# Visualize
viz = LONVisualizer()
viz.plot_3d(lon, output_path="landscape.png")
```

## Installation

```bash
pip install lonpy
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running in 5 minutes
- [Core Concepts](getting-started/concepts.md) - Understand LONs and fitness landscapes
- [API Reference](api/index.md) - Complete API documentation
