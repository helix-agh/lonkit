# Sampling Guide

This guide covers how to configure Basin-Hopping sampling for LON construction.

## Quick Start

The simplest way to create a LON:

```python
from lonpy import compute_lon, BasinHoppingSamplerConfig

config = BasinHoppingSamplerConfig(n_runs=20, seed=42)
lon = compute_lon(
    func=my_objective,
    dim=2,
    lower_bound=-5.0,
    upper_bound=5.0,
    config=config
)
```

## Configuration Options

For more control, use `BasinHoppingSamplerConfig`:

```python
from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig

config = BasinHoppingSamplerConfig(
    n_runs=30,                  # Number of independent runs
    n_iter_no_change=500,       # Max consecutive non-improving steps before stopping
    step_mode="percentage",     # "percentage" or "fixed"
    step_size=0.1,              # Perturbation magnitude
    coordinate_precision=5,     # Decimal places for node identification
    fitness_precision=None,     # Decimal places for fitness (None = full precision)
    bounded=True,               # Enforce domain bounds
    minimizer_method="L-BFGS-B",
    seed=42
)

sampler = BasinHoppingSampler(config)
lon = sampler.sample_to_lon(my_objective, domain)
```

## Parameters Explained

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | 100 | Number of independent Basin-Hopping runs |
| `n_iter_no_change` | 1000 | Max consecutive non-improving perturbations before stopping each run. At least one of `n_iter_no_change` or `max_iter` must be set. |
| `max_iter` | None | Max total perturbation steps per run. Use together with `n_iter_no_change` or alone. |
| `seed` | None | Random seed for reproducibility |

**Choosing `n_runs` and stopping criteria:**

- More runs = better coverage of the landscape
- `n_iter_no_change` counts *non-improving* consecutive steps - it is the primary stopping criterion per run
- `max_iter` caps total steps regardless of improvement - useful to bound computation time
- At least one of the two must be set; they can be combined

```python
# Wide coverage (recommended for unknown landscapes)
config = BasinHoppingSamplerConfig(n_runs=50, n_iter_no_change=200)

# Deep exploration (for complex local structure)
config = BasinHoppingSamplerConfig(n_runs=10, n_iter_no_change=1000)

# Hard cap on total iterations regardless of improvement
config = BasinHoppingSamplerConfig(n_runs=10, n_iter_no_change=None, max_iter=5000)
```

### Perturbation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_mode` | "fixed" | How to interpret step_size |
| `step_size` | 0.01 | Perturbation magnitude |
| `bounded` | True | Keep perturbations within domain |

**Step modes:**

```python
# Fixed: step_size is absolute distance
config = BasinHoppingSamplerConfig(
    step_mode="fixed",
    step_size=0.5  # Always perturb by ±0.5 in each dimension
)

# Percentage: step_size is fraction of domain range
config = BasinHoppingSamplerConfig(
    step_mode="percentage",
    step_size=0.1  # Perturb by ±10% of (upper - lower)
)
```

**Choosing step size:**

- Too small: Stays in same basin, misses transitions
- Too large: Jumps randomly, misses local structure
- Good starting point: 5-10% of domain range

### Precision Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coordinate_precision` | 5 | Decimal places for coordinate rounding and node identification (`None` = full double precision) |
| `fitness_precision` | None | Decimal places for fitness values (`None` = full double precision) |

**coordinate_precision** determines when two solutions are considered the same optimum:

```python
# High precision: More distinct nodes
config = BasinHoppingSamplerConfig(coordinate_precision=6)

# Low precision: More merging, fewer nodes
config = BasinHoppingSamplerConfig(coordinate_precision=2)

# Full precision: No rounding
config = BasinHoppingSamplerConfig(coordinate_precision=None)
```

**fitness_precision** controls rounding of fitness values:

```python
# Round fitness to 4 decimal places
config = BasinHoppingSamplerConfig(fitness_precision=4)

# Full double precision (default)
config = BasinHoppingSamplerConfig(fitness_precision=None)
```

### Local Minimizer Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minimizer_method` | "L-BFGS-B" | Scipy minimizer algorithm |
| `minimizer_options` | `{"ftol": 1e-07, "gtol": 0, "maxiter": 15000}` | Minimizer options |

```python
# Custom minimizer settings
config = BasinHoppingSamplerConfig(
    minimizer_method="L-BFGS-B",
    minimizer_options={
        "ftol": 1e-10,  # Tighter function tolerance
        "gtol": 1e-08,  # Tighter gradient tolerance
        "maxiter": 1000 # More iterations allowed
    }
)
```

## LON Construction Configuration

When constructing a LON from trace data, you can configure how duplicate nodes (nodes with multiple observed fitness values) are handled using `LONConfig`:

```python
from lonpy import LONConfig, BasinHoppingSampler, BasinHoppingSamplerConfig

lon_config = LONConfig(
    fitness_aggregation="min",       # How to resolve duplicate fitness values
    warn_on_duplicates=True,         # Warn when duplicates detected
    max_fitness_deviation=None,      # Error if deviation exceeds threshold
)

config = BasinHoppingSamplerConfig(n_runs=30, seed=42)
sampler = BasinHoppingSampler(config)
lon = sampler.sample_to_lon(my_objective, domain, lon_config=lon_config)
```

### Fitness Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `"min"` | Use minimum fitness (default) |
| `"max"` | Use maximum fitness |
| `"mean"` | Use average fitness |
| `"first"` | Use first occurrence |
| `"strict"` | Raise error if duplicates detected |

### Data Quality Checks

```python
# Strict mode: fail if any node has multiple fitness values
lon_config = LONConfig(fitness_aggregation="strict")

# Set a maximum allowed deviation
lon_config = LONConfig(max_fitness_deviation=0.01)
```

You can also pass `lon_config` to `compute_lon()`:

```python
from lonpy import compute_lon, LONConfig

lon = compute_lon(
    func=my_objective,
    dim=2,
    lower_bound=-5.0,
    upper_bound=5.0,
    lon_config=LONConfig(fitness_aggregation="mean"),
)
```

## Custom Initial Points

By default, Basin-Hopping starts each run from a random point sampled uniformly from the domain. You can provide custom starting points via `initial_points`:

```python
import numpy as np
from lonpy import compute_lon, BasinHoppingSampler, BasinHoppingSamplerConfig

# Generate custom initial points (must have shape (n_runs, dim))
n_runs = 30
dim = 2
initial_points = np.random.default_rng(0).uniform(-5.12, 5.12, size=(n_runs, dim))

# With compute_lon
config = BasinHoppingSamplerConfig(n_runs=n_runs, seed=42)
lon = compute_lon(
    func=my_objective,
    dim=dim,
    lower_bound=-5.12,
    upper_bound=5.12,
    initial_points=initial_points,
    config=config
)

# Or with BasinHoppingSampler
sampler = BasinHoppingSampler(config)
lon = sampler.sample_to_lon(my_objective, domain, initial_points=initial_points)
```

**Requirements:**

- Shape must be `(n_runs, dim)` — one point per run
- When `bounded=True`, all points must lie within the domain bounds

## Domain Specification

The domain is specified as a list of (lower, upper) tuples:

```python
# Same bounds for all dimensions
lon = compute_lon(func, dim=5, lower_bound=-5.0, upper_bound=5.0)

# Different bounds per dimension
domain = [
    (-5.0, 5.0),    # x1
    (0.0, 10.0),    # x2
    (-1.0, 1.0)     # x3
]
sampler = BasinHoppingSampler()
lon = sampler.sample_to_lon(func, domain)
```

## Accessing Raw Data

For custom analysis, access the raw trace data:

```python
sampler = BasinHoppingSampler(config)
trace_df, raw_records = sampler.sample(func, domain)

# trace_df columns: [run, fit1, node1, fit2, node2]
print(trace_df.head())

# raw_records contains detailed iteration data
for record in raw_records[:5]:
    print(f"Run {record['run']}, Iter {record['iteration']}")
    print(f"  Current: {record['current_f']:.4f}")
    print(f"  New: {record['new_f']:.4f}")
    print(f"  Accepted: {record['accepted']}")
```

## Progress Monitoring

Track sampling progress with a callback:

```python
def progress(run, total):
    print(f"Run {run}/{total}")

sampler = BasinHoppingSampler(config)
lon = sampler.sample_to_lon(func, domain, progress_callback=progress)
```

## Best Practices

### For Standard Test Functions

```python
# Rastrigin, Ackley, etc. with known bounds
config = BasinHoppingSamplerConfig(
    n_runs=30,
    n_iter_no_change=500,
    step_mode="percentage",
    step_size=0.1,
    coordinate_precision=4,
    seed=42
)
```

### For Unknown Functions

```python
# Start with wider exploration
config = BasinHoppingSamplerConfig(
    n_runs=50,
    n_iter_no_change=200,
    step_mode="percentage",
    step_size=0.15,            # Larger steps initially
    coordinate_precision=3,    # Coarser grouping
)

# Refine based on initial results
```

### For High-Dimensional Problems

```python
# More runs needed for coverage
config = BasinHoppingSamplerConfig(
    n_runs=100,
    n_iter_no_change=500,
    step_mode="percentage",
    step_size=0.05,  # Smaller relative steps
)
```

## Next Steps

- [Analysis Guide](analysis.md) - Interpret your LON metrics
- [Visualization Guide](visualization.md) - Create plots
- [API Reference](../api/sampling.md) - Full API documentation
