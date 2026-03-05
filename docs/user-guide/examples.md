# Examples

Complete examples demonstrating lonpy's capabilities.

## Basic LON Analysis

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer, BasinHoppingSamplerConfig

# Define the Rastrigin function
def rastrigin(x):
    """Highly multimodal test function."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Build the LON
config = BasinHoppingSamplerConfig(n_runs=30, n_iter_no_change=500, seed=42)
lon = compute_lon(
    func=rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    config=config
)

# Analyze
print("=== LON Analysis ===")
print(f"Local optima: {lon.n_vertices}")
print(f"Transitions: {lon.n_edges}")
print(f"Best fitness: {lon.best_fitness}")

metrics = lon.compute_metrics()
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(lon, output_path="rastrigin_lon.png", seed=42)
viz.plot_3d(lon, output_path="rastrigin_3d.png", seed=42)
```

## Comparing Multiple Functions

```python
import numpy as np
import pandas as pd
from lonpy import compute_lon, BasinHoppingSamplerConfig

# Test functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

# Configuration
functions = {
    "Sphere": (sphere, -5.0, 5.0, 0.0),
    "Rastrigin": (rastrigin, -5.12, 5.12, 0.0),
    "Ackley": (ackley, -5.0, 5.0, 0.0),
    "Rosenbrock": (rosenbrock, -5.0, 10.0, 0.0),
}

# Analyze each
results = []
for name, (func, lb, ub, optimal) in functions.items():
    print(f"Analyzing {name}...")

    config = BasinHoppingSamplerConfig(n_runs=30, n_iter_no_change=500, seed=42)
    lon = compute_lon(
        func=func,
        dim=2,
        lower_bound=lb,
        upper_bound=ub,
        config=config
    )

    metrics = lon.compute_metrics(known_best=optimal)
    cmlon = lon.to_cmlon()
    cmlon_metrics = cmlon.compute_metrics(known_best=optimal)

    results.append({
        "Function": name,
        "Optima": lon.n_vertices,
        "Funnels": metrics['n_funnels'],
        "Global Funnels": metrics['n_global_funnels'],
        "Global Strength": f"{metrics['global_strength']:.1%}",
        "Sink Strength": f"{metrics['sink_strength']:.1%}",

        "Global Funnel %": f"{cmlon_metrics['global_funnel_proportion']:.1%}",
    })

# Display results
df = pd.DataFrame(results)
print("\n=== Comparison ===")
print(df.to_string(index=False))
```

## Detailed CMLON Analysis

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer, BasinHoppingSamplerConfig

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Build LON
config = BasinHoppingSamplerConfig(n_runs=50, seed=42)
lon = compute_lon(
    rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    config=config
)

# Convert to CMLON
cmlon = lon.to_cmlon()

print("=== LON vs CMLON ===")
print(f"LON vertices:   {lon.n_vertices}")
print(f"CMLON vertices: {cmlon.n_vertices}")
print(f"Compression:    {1 - cmlon.n_vertices/lon.n_vertices:.1%}")

print("\n=== CMLON Structure ===")
print(f"Total sinks:  {len(cmlon.get_sinks())}")
print(f"Global sinks: {len(cmlon.get_global_sinks())}")
print(f"Local sinks:  {len(cmlon.get_local_sinks())}")

# Metrics
cmlon_metrics = cmlon.compute_metrics()
print("\n=== CMLON Metrics ===")
print(f"Global funnel proportion: {cmlon_metrics['global_funnel_proportion']:.1%}")
print(f"Global strength: {cmlon_metrics['global_strength']:.1%}")
print(f"Sink strength: {cmlon_metrics['sink_strength']:.1%}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(cmlon, output_path="cmlon_2d.png", seed=42)
viz.plot_3d(cmlon, output_path="cmlon_3d.png", seed=42)
```

## Custom Sampling Configuration

```python
import numpy as np
from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

def schwefel(x):
    """Schwefel function - deceptive with distant global optimum."""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Custom configuration for challenging function
config = BasinHoppingSamplerConfig(
    n_runs=100,              # More runs for coverage
    n_iter_no_change=300,    # Stop after 300 non-improving steps
    step_mode="percentage",
    step_size=0.15,          # Larger steps for this landscape
    coordinate_precision=3,  # Coarser grouping
    bounded=True,
    minimizer_method="L-BFGS-B",
    minimizer_options={
        "ftol": 1e-08,
        "gtol": 1e-06,
    },
    seed=42
)

# Sample
domain = [(-500.0, 500.0), (-500.0, 500.0)]
sampler = BasinHoppingSampler(config)

def progress(run, total):
    if run % 10 == 0:
        print(f"Progress: {run}/{total}")

lon = sampler.sample_to_lon(schwefel, domain, progress_callback=progress)

# Analyze
print(f"\nFound {lon.n_vertices} local optima")
print(f"Best fitness: {lon.best_fitness}")

# Known optimum at x = (420.9687, ...) with f(x) ≈ 0
metrics = lon.compute_metrics()
print(f"Funnels: {metrics['n_funnels']}")
print(f"Global strength: {metrics['global_strength']:.1%}")
print(f"Sink strength: {metrics['sink_strength']:.1%}")
```

## Accessing Raw Trace Data

```python
import numpy as np
import pandas as pd
from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig

def sphere(x):
    return np.sum(x**2)

config = BasinHoppingSamplerConfig(n_runs=5, n_iter_no_change=100, seed=42)
sampler = BasinHoppingSampler(config)

domain = [(-5.0, 5.0), (-5.0, 5.0)]
trace_df, raw_records = sampler.sample(sphere, domain)

# Trace data for LON construction
print("=== Trace Data ===")
print(trace_df.head(10))
print(f"\nTotal transitions: {len(trace_df)}")

# Raw iteration data
print("\n=== Sample Raw Records ===")
for i, rec in enumerate(raw_records[:5]):
    print(f"Run {rec['run']}, Iter {rec['iteration']}:")
    print(f"  From: {rec['current_f']:.6f} -> To: {rec['new_f']:.6f}")
    print(f"  Accepted: {rec['accepted']}")

# Analyze acceptance rate
accepted = sum(r['accepted'] for r in raw_records)
total = len(raw_records)
print(f"\nAcceptance rate: {accepted/total:.1%}")
```

## Working with the Graph Directly

```python
import numpy as np
from lonpy import compute_lon, BasinHoppingSamplerConfig

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

lon = compute_lon(rastrigin, dim=2, lower_bound=-5.12, upper_bound=5.12,
                  config=BasinHoppingSamplerConfig(n_runs=30, seed=42))

# Access igraph object
g = lon.graph

print("=== Graph Properties ===")
print(f"Vertices: {g.vcount()}")
print(f"Edges: {g.ecount()}")
print(f"Density: {g.density():.4f}")

# Find most visited nodes
counts = g.vs["Count"]
top_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:5]

print("\n=== Most Visited Optima ===")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. Node {g.vs[idx]['name']}: {counts[idx]} visits, fitness={g.vs[idx]['Fitness']}")

# Analyze edge weights
if g.ecount() > 0:
    edge_weights = g.es["Count"]
    print(f"\n=== Edge Statistics ===")
    print(f"Total edges: {len(edge_weights)}")
    print(f"Total transitions: {sum(edge_weights)}")
    print(f"Max edge weight: {max(edge_weights)}")
    print(f"Mean edge weight: {np.mean(edge_weights):.2f}")
```

## Batch Analysis Script

```python
import numpy as np
import json
from pathlib import Path
from lonpy import compute_lon, LONVisualizer, BasinHoppingSamplerConfig

def analyze_function(name, func, bounds, output_dir):
    """Analyze a function and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build LON
    config = BasinHoppingSamplerConfig(n_runs=30, n_iter_no_change=500, seed=42)
    lon = compute_lon(
        func=func,
        dim=2,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        config=config
    )

    # Compute metrics
    lon_metrics = lon.compute_metrics()
    cmlon = lon.to_cmlon()
    cmlon_metrics = cmlon.compute_metrics()

    # Save metrics
    results = {
        "function": name,
        "lon": {
            "n_vertices": lon.n_vertices,
            "n_edges": lon.n_edges,
            "best_fitness": lon.best_fitness,
            **lon_metrics
        },
        "cmlon": {
            "n_vertices": cmlon.n_vertices,
            "n_edges": cmlon.n_edges,
            **cmlon_metrics
        }
    }

    with open(output_dir / f"{name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    viz = LONVisualizer()
    viz.visualize_all(lon, output_dir / name, seed=42)

    return results

# Example usage
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

results = analyze_function("rastrigin", rastrigin, (-5.12, 5.12), "./analysis")
print(json.dumps(results, indent=2))
```

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Core Concepts](../getting-started/concepts.md) - Understand LON theory
