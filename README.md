<div align="center">
    <img src="https://raw.githubusercontent.com/helix-agh/lonkit/main/docs/assets/icon.jpg" alt="lonkit: Local Optima Networks visualization in Python" width="800">
</div>

<h1 align="center">lonkit: Local Optima Networks in Python</h1>

[![PyPI version](https://badge.fury.io/py/lonkit.svg)](https://pypi.org/project/lonkit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-helix--agh.github.io%2Flonkit-indigo.svg)](https://helix-agh.github.io/lonkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ujl48ffgHg9ck1Hueh59s65OR3Q3BG99?usp=sharing)

**lonkit** is a Python library for constructing, analyzing, and visualizing **Local Optima Networks (LONs)** for both continuous and discrete optimization problems. LONs provide a powerful way to understand the structure of fitness landscapes, revealing how local optima are connected and how difficult it may be to find global optima.

**Documentation:** [helix-agh.github.io/lonkit](https://helix-agh.github.io/lonkit/)

## Features

- **Basin-Hopping Sampling**: Efficient exploration of continuous fitness landscapes using configurable Basin-Hopping
- **Iterated Local Search**: Discrete LON construction via ILS with built-in problems (Number Partitioning, NK Landscape, OneMax)
- **LON Construction**: Automatic construction of Local Optima Networks from sampling data
- **CMLON Support**: Compressed Monotonic LONs for cleaner landscape analysis
- **Rich Metrics**: Compute landscape metrics including funnel analysis and neutrality
- **Beautiful Visualizations**: 2D and 3D plots with support for animated GIFs

## Installation

```bash
pip install lonkit
```

Exporting 3D plots to images or GIFs also requires Chrome/Chromium. If it is not
installed, run `kaleido_get_chrome` after installing lonkit. See the
[installation guide](https://helix-agh.github.io/lonkit/getting-started/installation/).

Or install from source:

```bash
git clone https://github.com/helix-agh/lonkit.git
cd lonkit
pip install -e .
```

## Quick Start

```python
import numpy as np
from lonkit import compute_lon, LONVisualizer, BasinHoppingSamplerConfig

# Define an objective function
def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Construct the LON
config = BasinHoppingSamplerConfig(
    n_runs=20,
    n_iter_no_change=500,
    seed=42
)
lon = compute_lon(
    rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    config=config
)

metrics = lon.compute_metrics()
print(f"Number of funnels: {metrics['n_funnels']}")
print(f"Global funnels: {metrics['n_global_funnels']}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(lon, output_path="lon_2d.png")
viz.plot_3d(lon, output_path="lon_3d.png")
```

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/helix-agh/lonkit/main/docs/assets/readme/rastrigin_lon_2d.png" alt="2D plot of the Rastrigin LON" width="400"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/helix-agh/lonkit/main/docs/assets/readme/rastrigin_lon_3d.png" alt="3D plot of the Rastrigin LON with fitness on the Z-axis" width="400"></td>
  </tr>
  <tr>
    <td align="center"><code>plot_2d(lon)</code></td>
    <td align="center"><code>plot_3d(lon)</code>: fitness on the Z-axis</td>
  </tr>
</table>

Nodes are local optima and edges are transitions between them, with node size and edge width growing with how often they were visited.

### Compressed Monotonic LONs (CMLONs)

CMLONs are a compressed representation where nodes with equal fitness that are connected get merged. This provides a cleaner view of the landscape's funnel structure.

```python
# Convert LON to CMLON
cmlon = lon.to_cmlon()

# Analyze CMLON-specific metrics
cmlon_metrics = cmlon.compute_metrics()
```

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/helix-agh/lonkit/main/docs/assets/readme/rastrigin_lon_2d.png" alt="Rastrigin LON" width="400"></td>
    <td align="center"><img src="https://raw.githubusercontent.com/helix-agh/lonkit/main/docs/assets/readme/rastrigin_cmlon_2d.png" alt="Rastrigin CMLON" width="400"></td>
  </tr>
  <tr>
    <td align="center">LON: 45 optima</td>
    <td align="center">CMLON: 39 optima</td>
  </tr>
</table>

### Custom Sampling Configuration

```python
from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig

config = BasinHoppingSamplerConfig(
    n_runs=50,                   # Number of independent runs
    n_iter_no_change=1000,       # Stop after this many consecutive non-improving perturbations
    step_size=0.05,              # Perturbation size
    step_mode="percentage",      # "percentage" or "fixed"
    coordinate_precision=4,      # Precision for identifying optima
    fitness_precision=None,      # Precision for fitness values (None = full double)
    seed=42                      # For reproducibility
)

sampler = BasinHoppingSampler(config)

# Define search domain
domain = [(-5.12, 5.12), (-5.12, 5.12)]

# Run sampling
result = sampler.sample(rastrigin, domain)
lon = sampler.sample_to_lon(result)
```

## Discrete Optimization Problems

lonkit also supports Local Optima Networks for discrete optimization problems using Iterated Local Search (ILS).

### Quick Start (Discrete)

```python
from lonkit import NumberPartitioning, ILSSampler, ILSSamplerConfig, LONVisualizer

# Define problem instance
problem = NumberPartitioning(n=20, k=0.5, instance_seed=1)

# Configure and run ILS sampling
config = ILSSamplerConfig(n_runs=50, n_iter_no_change=100, seed=42)
sampler = ILSSampler(config)
result = sampler.sample(problem)

# Build LON and CMLON
lon = sampler.sample_to_lon(result)
cmlon = lon.to_cmlon()

# Compute metrics
print(lon.compute_metrics())
print(cmlon.compute_metrics())

# Visualize
viz = LONVisualizer()
viz.plot_2d(cmlon, output_path="npp_cmlon.png")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/helix-agh/lonkit/main/docs/assets/readme/npp_cmlon_2d.png" alt="CMLON of a Number Partitioning instance with many global funnels and one local funnel" width="500">
</p>

Unlike Rastrigin, this Number Partitioning instance has many separate funnels: most of them end in a global optimum (red), and one ends in a worse local sink (blue).

### Custom Discrete Problems

You can define your own discrete problem by subclassing `DiscreteProblem`. For bitstring problems, inherit from `BitstringProblem` instead; it provides `random_solution()`, `local_search()`, `perturb()`, and `solution_id()` out of the box. You only need to implement `evaluate()`.

## Documentation

The full documentation is at [helix-agh.github.io/lonkit](https://helix-agh.github.io/lonkit/):

- [Installation](https://helix-agh.github.io/lonkit/getting-started/installation/)
- [Quick Start](https://helix-agh.github.io/lonkit/getting-started/quickstart/)
- [Core Concepts: LONs and fitness landscapes](https://helix-agh.github.io/lonkit/getting-started/concepts/)
- [Discrete Problems](https://helix-agh.github.io/lonkit/user-guide/discrete/)
- [API Reference](https://helix-agh.github.io/lonkit/api/)

## References

- G. Ochoa, M. Tomassini, S. Vérel, C. Darabos. *A Study of NK Landscapes' Basins and Local Optima Networks.* GECCO 2008.
- G. Ochoa, N. Veerapen, F. Daolio, M. Tomassini. *Understanding Phase Transitions with Local Optima Networks: Number Partitioning as a Case Study.* EvoCOP 2017.
- J. Adair, G. Ochoa, K. M. Malan. *Local Optima Networks for Continuous Fitness Landscapes.* GECCO 2019 Companion.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
