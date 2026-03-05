# Analysis Guide

This guide explains how to analyze LONs and interpret the computed metrics.

## Computing Metrics

Both LON and CMLON provide a `compute_metrics()` method:

```python
from lonpy import compute_lon, BasinHoppingSamplerConfig

lon = compute_lon(func, dim=2, lower_bound=-5, upper_bound=5,
                  config=BasinHoppingSamplerConfig(n_runs=30))

# LON metrics
lon_metrics = lon.compute_metrics()
print(lon_metrics)

# CMLON metrics
cmlon = lon.to_cmlon()
cmlon_metrics = cmlon.compute_metrics()
print(cmlon_metrics)
```

## LON Metrics

### n_optima

**Number of local optima discovered.**

```python
n_optima = lon_metrics['n_optima']
print(f"Found {n_optima} local optima")
```

**Interpretation:**

- Higher values indicate more multimodal landscapes
- Compare across functions to assess relative complexity
- Depends on sampling thoroughness and coordinate_precision

### n_funnels

**Number of sink nodes (nodes with no outgoing edges).**

```python
n_funnels = lon_metrics['n_funnels']
print(f"Landscape has {n_funnels} funnels")
```

**Interpretation:**

- 1 funnel = single-funnel landscape (easier to optimize)
- Multiple funnels = competing basins (harder to optimize)
- Sinks are "endpoints" where Basin-Hopping cannot escape

### n_global_funnels

**Number of sinks at the global optimum fitness.**

```python
n_global = lon_metrics['n_global_funnels']
print(f"{n_global} funnels lead to global optimum")
```

**Interpretation:**

- Ideally equals n_funnels (all paths lead to global)
- If n_global < n_funnels, some searches get trapped
- Multiple global funnels may indicate symmetry in the landscape

### neutral

**Proportion of nodes with equal-fitness connections.**

```python
neutral = lon_metrics['neutral']
print(f"Neutrality: {neutral:.1%}")
```

**Interpretation:**

- 0% = No flat regions, all transitions change fitness
- High % = Many plateaus or degenerate optima
- Affects how CMLON compression works

### global_strength

**Proportion of global optima incoming strength to total incoming strength of all nodes.**

```python
global_strength = lon_metrics['global_strength']
print(f"Global strength: {global_strength:.1%}")
```

**Interpretation:**

- 100% = All transitions flow toward global optimum
- Low % = Most flow diverted to suboptimal nodes
- Key indicator of optimization difficulty

### sink_strength

**Proportion of global sinks incoming strength to incoming strength of all sink nodes.**

```python
sink_strength = lon_metrics['sink_strength']
print(f"Sink strength: {sink_strength:.1%}")
```

**Interpretation:**

- 100% = All sink-directed flow reaches global sinks
- Low % = Most sink flow goes to local (suboptimal) sinks
- Focuses specifically on competition between sinks

## Performance Metrics

In addition to network topology metrics, `compute_metrics()` returns performance metrics based on the sampling runs:

### success

**Proportion of sampling runs that reached the global optimum.**

```python
success = lon_metrics['success']
print(f"Success rate: {success:.1%}")
```

**Interpretation:**

- 100% = Every run found the global optimum
- Low % = Many runs got trapped in local optima
- Useful for comparing algorithm effectiveness across landscapes

### deviation

**Mean absolute deviation from the global optimum across all runs.**

```python
deviation = lon_metrics['deviation']
print(f"Mean deviation: {deviation:.6f}")
```

**Interpretation:**

- 0.0 = All runs converged to the global optimum
- Higher values indicate runs ending far from the global optimum
- Complements `success` by measuring "how far off" unsuccessful runs are

### Separating Network and Performance Metrics

You can also compute them separately:

```python
# Only network topology metrics (n_optima, n_funnels, etc.)
network_metrics = lon.compute_network_metrics()

# Only performance metrics (success, deviation)
performance_metrics = lon.compute_performance_metrics()

# Both combined (equivalent to compute_metrics())
all_metrics = lon.compute_metrics()
```

## CMLON-Specific Metrics

CMLON computes additional metrics:

### global_funnel_proportion

**Proportion of nodes that can reach a global optimum.**

```python
gfp = cmlon_metrics['global_funnel_proportion']
print(f"Global funnel proportion: {gfp:.1%}")
```

**Interpretation:**

- 100% = Every optimum can reach the global (easy landscape)
- Low % = Many optima trapped in local funnels
- Better measure of "escapability" than n_funnels

### neutral (in CMLON)

For CMLON, neutral measures compression ratio:

```python
compression = cmlon_metrics['neutral']
print(f"Compression: {compression:.1%} of nodes merged")
```

## Comparing LON vs CMLON

```python
print("=== LON ===")
print(f"Nodes: {lon.n_vertices}")
print(f"Edges: {lon.n_edges}")

print("\n=== CMLON ===")
print(f"Nodes: {cmlon.n_vertices}")
print(f"Edges: {cmlon.n_edges}")

print(f"\nCompression: {1 - cmlon.n_vertices/lon.n_vertices:.1%}")
```

## Accessing Graph Properties

### Vertex Properties

```python
# All vertex fitness values
fitnesses = lon.vertex_fitness

# Best fitness found
best = lon.best_fitness

# Vertex visit counts
counts = lon.vertex_count

# Vertex names (hash strings)
names = lon.vertex_names
```

### Finding Special Nodes

```python
# Sink nodes (no outgoing edges)
sinks = lon.get_sinks()
print(f"Sink indices: {sinks}")

# For CMLON: separate global and local sinks
global_sinks = cmlon.get_global_sinks()
local_sinks = cmlon.get_local_sinks()
```

### Working with the Graph

The underlying graph is accessible via the `graph` attribute (python-igraph):

```python
import igraph as ig

# Get the igraph Graph object
g = lon.graph

# Graph properties
print(f"Vertices: {g.vcount()}")
print(f"Edges: {g.ecount()}")
print(f"Directed: {g.is_directed()}")

# Edge list
edges = g.get_edgelist()

# Vertex attributes
all_fitness = g.vs["Fitness"]
all_counts = g.vs["Count"]

# Edge attributes
edge_weights = g.es["Count"]

# Compute standard graph metrics
print(f"Density: {g.density():.4f}")
print(f"Components: {len(g.components(mode='weak'))}")
```

## Using Known Global Optimum

If you know the true global optimum, pass it to `compute_metrics()`:

```python
# Rastrigin global optimum is 0
metrics = lon.compute_metrics(known_best=0)

# Check if we found it
if lon.best_fitness == 0:
    print("Global optimum found!")
else:
    print(f"Best found: {lon.best_fitness} (gap: {lon.best_fitness - 0})")
```

## Landscape Classification

Use metrics to classify landscape difficulty:

```python
def classify_landscape(lon):
    metrics = lon.compute_metrics()
    cmlon = lon.to_cmlon()
    cmlon_metrics = cmlon.compute_metrics()

    if metrics['n_funnels'] == 1:
        return "Easy: Single-funnel landscape"
    elif cmlon_metrics['global_funnel_proportion'] > 0.8:
        return "Moderate: Multiple funnels but well-connected"
    elif metrics['global_strength'] > 0.5:
        return "Moderate: Good flow to global optimum"
    else:
        return "Hard: Multiple competing funnels"

print(classify_landscape(lon))
```

## Comparing Multiple Functions

```python
import numpy as np
from lonpy import compute_lon, BasinHoppingSamplerConfig

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

functions = {
    "Sphere": (sphere, -5, 5),
    "Rastrigin": (rastrigin, -5.12, 5.12),
    "Rosenbrock": (rosenbrock, -5, 10),
}

results = {}
for name, (func, lb, ub) in functions.items():
    lon = compute_lon(func, dim=2, lower_bound=lb, upper_bound=ub,
                      config=BasinHoppingSamplerConfig(n_runs=30, seed=42))
    results[name] = lon.compute_metrics()
    results[name]['n_vertices'] = lon.n_vertices

# Compare
import pandas as pd
df = pd.DataFrame(results).T
print(df)
```

## Next Steps

- [Visualization Guide](visualization.md) - Create plots of your analysis
- [Examples](examples.md) - Complete analysis workflows
- [API Reference](../api/lon.md) - Full LON/CMLON API
