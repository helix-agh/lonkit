---
description: Configuration and usage of Basin-Hopping and Iterated Local Search sampling for constructing Local Optima Networks. Perturbation, step size, and stopping condition controls.
---

# Sampling Guide

This guide covers Basin-Hopping sampling for continuous problems and Iterated Local Search (ILS) for discrete problems.

## Quick Start

The simplest way to create a LON:

```python
from lonkit import compute_lon, BasinHoppingSamplerConfig

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
from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig

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
result = sampler.sample(my_objective, domain)
lon = sampler.sample_to_lon(result)
```

## Parameters Explained

### Sampling Parameters

| Parameter          | Default | Description                                                                                                                                                                                             |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_runs`           | 100     | Number of independent Basin-Hopping runs                                                                                                                                                                |
| `n_iter_no_change` | 250     | Max consecutive non-improving perturbations before stopping each run. At least one of `n_iter_no_change` or `max_iter` must be set.                                                                     |
| `max_iter`         | None    | Max total perturbation steps per run. Use together with `n_iter_no_change` or alone.                                                                                                                    |
| `seed`             | None    | Random seed for reproducibility                                                                                                                                                                         |
| `n_jobs`           | 1       | Maximum number of runs executed in parallel; `1` runs sequentially, `-1` uses all CPUs. See [Parallel Execution](#parallel-execution).                                                                  |

**Choosing `n_runs` and stopping criteria:**

- More runs = better coverage of the landscape
- `n_iter_no_change` counts _non-improving_ consecutive steps - it is the primary stopping criterion per run
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

| Parameter   | Default      | Description                      |
| ----------- | ------------ | -------------------------------- |
| `step_mode` | "percentage" | How to interpret step_size       |
| `step_size` | 0.1          | Perturbation magnitude           |
| `bounded`   | True         | Keep perturbations within domain |

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

> **Tip:** Use `StepSizeEstimator` to automatically find the optimal step size for your problem. See the [API Reference](../api/step_size.md) for details.

### Precision Settings

| Parameter              | Default | Description                                                                                     |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `coordinate_precision` | 5       | Decimal places for coordinate rounding and node identification (`None` = full double precision) |
| `fitness_precision`    | None    | Decimal places for fitness values (`None` = full double precision)                              |

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

| Parameter           | Default    | Description               |
| ------------------- | ---------- | ------------------------- |
| `minimizer_method`  | "L-BFGS-B" | Scipy minimizer algorithm |
| `minimizer_options` | `None`     | Minimizer options         |

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

### Failed Local Searches

A local search fails when `minimize` raises `ValueError` or returns a non-finite
fitness (`nan` or `inf`). Such a result never enters the LON:

- if the initial local search of a run fails, the whole run fails and is left out
  of the trace and the performance metrics;
- if a local search after a perturbation fails, it is rejected and counts as a
  non-improving iteration.

Results with `success=False` but a finite fitness are used as local optima. On
non-smooth functions such as Ackley, L-BFGS-B often reports `ABNORMAL` at genuine
optima.

Each failed local search emits a warning with the run and iteration (for
perturbations), the starting point, and the reason for failure.

## Discrete Sampling with ILS

Independent ILS runs can use multiple processes via `n_jobs`, with the same
semantics as `BasinHoppingSampler` (see [Parallel Execution](#parallel-execution)):

```python
from lonkit import ILSSampler, ILSSamplerConfig, NumberPartitioning

problem = NumberPartitioning(n=20, k=0.5, instance_seed=1)
sampler = ILSSampler(ILSSamplerConfig(n_runs=20, max_iter=100, seed=42, n_jobs=2))
result = sampler.sample(problem)
lon = sampler.sample_to_lon(result)
```

When running in parallel, custom problems must be serializable by joblib and must
not hold mutable search state. All runs use the supplied problem's data; workers
do not regenerate instances. Custom problems must also use the supplied RNG for
all search randomness, otherwise results are not reproducible.

**Seeded ILS results change compared with earlier versions**, which shared one
RNG across all runs. Progress callbacks are now called after each run completes
rather than before it starts. The `ILSResult` format and `sample_to_lon()` usage
remain unchanged.

## Parallel Execution

Both samplers run their `n_runs` independent runs in parallel according to the
`n_jobs` config parameter, which follows
[joblib semantics](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html):

| `n_jobs`      | Behaviour                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------ |
| `1` (default) | Runs execute sequentially in the main process, without workers                             |
| `N > 1`       | Up to `N` worker processes                                                                 |
| `-1`          | One worker process per CPU                                                                 |
| `-N`          | Up to `max(1, cpu_count - N + 1)` worker processes, e.g. `-2` leaves one CPU free          |
| `None`        | Value from an enclosing `joblib.parallel_config(n_jobs=...)` context, otherwise `1`        |
| `0`           | Invalid, raises `ValueError`                                                               |

Whenever the effective number of jobs is `1`, runs execute directly in the main
process. Otherwise joblib worker processes are used, so the objective function
(or discrete problem) must be serializable by joblib. For short runs, process
startup and serialization overhead can outweigh the benefit of parallelism.
Worker errors propagate to the caller. `BasinHoppingSampler` also re-emits
warnings raised in workers (such as [failed local searches](#failed-local-searches))
in the main process.

#### Reproducibility

Each run gets its own RNG from
`SeedSequence(seed).spawn(n_runs)`, whatever the value of `n_jobs`. With a fixed
`seed`, the resulting `trace_df` and LON are identical across `n_jobs` values.

#### Ordering and progress callbacks

`progress_callback(completed_runs, total_runs)`
is always called in the main process, once per completed run, with counts from `1`
through `n_runs`. The two samplers differ in how parallel results are collected:

| Sampler               | Results collected     | `raw_records` order across `n_jobs` | Callback timing                                     |
| --------------------- | --------------------- | ----------------------------------- | --------------------------------------------------- |
| `BasinHoppingSampler` | As runs finish        | May differ (`trace_df` is sorted)   | As soon as any run finishes                         |
| `ILSSampler`          | In run order          | Identical                           | In run order; a slow early run delays later updates |

## LON Construction Configuration

When constructing a LON from trace data, you can configure how duplicate nodes (nodes with multiple observed fitness values) are handled using `LONConfig`:

```python
from lonkit import LONConfig, BasinHoppingSampler, BasinHoppingSamplerConfig

lon_config = LONConfig(
    fitness_aggregation="min",       # How to resolve duplicate fitness values
    warn_on_duplicates=True,         # Warn when duplicates detected
    max_fitness_deviation=None,      # Error if deviation exceeds threshold
)

config = BasinHoppingSamplerConfig(n_runs=30, seed=42)
sampler = BasinHoppingSampler(config)
result = sampler.sample(my_objective, domain)
lon = sampler.sample_to_lon(result, lon_config=lon_config)
```

### Fitness Aggregation Strategies

| Strategy   | Description                        |
| ---------- | ---------------------------------- |
| `"min"`    | Use minimum fitness (default)      |
| `"max"`    | Use maximum fitness                |
| `"mean"`   | Use average fitness                |
| `"first"`  | Use first occurrence               |
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
from lonkit import compute_lon, LONConfig

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
from lonkit import compute_lon, BasinHoppingSampler, BasinHoppingSamplerConfig

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
result = sampler.sample(my_objective, domain, initial_points=initial_points)
lon = sampler.sample_to_lon(result)
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
result = sampler.sample(func, domain)
lon = sampler.sample_to_lon(result)
```

## Accessing Raw Data

For custom analysis, access the raw trace data:

```python
sampler = BasinHoppingSampler(config)
result = sampler.sample(func, domain)

# trace_df columns: [run, fit1, node1, fit2, node2]
print(result.trace_df.head())

# raw_records contains detailed iteration data
for record in result.raw_records[:5]:
    print(f"Run {record['run']}, Iter {record['iteration']}")
    print(f"  Current: {record['current_f']:.4f}")
    print(f"  New: {record['new_f']:.4f}")
    print(f"  Accepted: {record['accepted']}")
```

## Loading External Traces

`LON.from_trace_data()` accepts traces produced by any sampler, including ones implemented outside of Python. Write the transitions to a file and load them as a `DataFrame` with exactly these columns, in any order:

| Column  | Description                                     |
| ------- | ----------------------------------------------- |
| `run`   | Integer run number                              |
| `fit1`  | Fitness of the source local optimum             |
| `node1` | String identifier of the source local optimum   |
| `fit2`  | Fitness of the target local optimum             |
| `node2` | String identifier of the target local optimum   |

Each row is one **accepted** transition between local optima. Within a run, rows must be in chronological order: the fitness in the last row of a run is taken as the run's final result. The same local optimum must always get the same identifier.

A run that never accepts a move after its initial local search should still get one row, from its local optimum to itself (`node1 == node2`). Otherwise its optimum is missing from the LON and the run is left out of `success` and `deviation`. Self-loops are dropped from the graph, so such rows add a vertex but no edges. The built-in samplers record these runs this way by default.

Both `BasinHoppingSamplerConfig` and `ILSSamplerConfig` accept
`preserve_runs_without_transitions=False` to restore the earlier behavior:

```python
config = BasinHoppingSamplerConfig(preserve_runs_without_transitions=False)
```

This can be useful when reproducing earlier results or analyzing a network built
only from accepted transitions. Runs without accepted moves are then omitted
from both the trace and the performance metrics: `success` and `deviation`
describe only the retained runs, not all completed runs. The option does not
remove all isolated vertices: accepted moves to the same node still produce
trace rows, whose self-loops are removed when building the graph. Search behavior
and accepted transitions are unchanged.

```python
import pandas as pd
from lonkit import LON, LONConfig

trace = pd.read_csv("trace.csv", dtype={"node1": str, "node2": str})
lon = LON.from_trace_data(trace, config=LONConfig(minimize=True))
```

!!! warning "Leading zeros in node identifiers"
    Without `dtype={"node1": str, "node2": str}`, `pd.read_csv()` parses identifiers such as bitstrings as integers and drops their leading zeros (`"0110"` becomes `110`), which could merge different local optima into one node. `LON.from_trace_data()` therefore rejects non-string node identifiers.

The trace is validated before the LON is built:

- a `ValueError` is raised for malformed data, e.g. missing or extra columns, missing values, non-integer run numbers, non-numeric or infinite fitness values, or non-string node identifiers;
- a `UserWarning` is emitted when a trajectory is broken, i.e. within a run `node2` of a row differs from `node1` of the next row. This usually means that rows are out of order or that rejected moves were included in the trace.

## Progress Monitoring

Track sampling progress with a callback or the `verbose` flag:

```python
# Using verbose flag (prints progress bar)
result = sampler.sample(func, domain, verbose=True)

# Using a custom callback
def progress(run, total):
    print(f"Run {run}/{total}")

sampler = BasinHoppingSampler(config)
result = sampler.sample(func, domain, progress_callback=progress)
lon = sampler.sample_to_lon(result)
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
