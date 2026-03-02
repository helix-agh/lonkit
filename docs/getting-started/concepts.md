# Core Concepts

This page explains the key concepts behind Local Optima Networks and how lonpy implements them.

## Fitness Landscapes

A **fitness landscape** is a conceptual model used to visualize the relationship between solutions and their quality (fitness). For optimization problems:

- The **search space** consists of all possible solutions
- The **fitness function** assigns a quality value to each solution
- **Neighbors** are solutions reachable through small modifications

Think of it as a terrain where:

- Position represents a solution
- Elevation represents fitness (lower = better for minimization)
- Walking represents searching for better solutions

## Local Optima

A **local optimum** is a solution that cannot be improved by small changes. In continuous optimization, this means:

$$\nabla f(x^*) = 0 \quad \text{and} \quad \nabla^2 f(x^*) \succeq 0$$

Local optima are important because:

- **Gradient-based methods** get stuck at local optima
- **Multimodal functions** have many local optima
- The **global optimum** is the best local optimum

## Basin-Hopping

**Basin-Hopping** is a global optimization algorithm that escapes local optima through:

1. **Local minimization**: Find nearest local optimum
2. **Perturbation**: Random step to escape the current basin
3. **Acceptance**: Move to new optimum if it is equal or better 

This creates a trajectory through the space of local optima:

```
Local Opt A → (perturb) → Local Opt B → (perturb) → Local Opt C → ...
```

lonpy records these transitions to build the LON.

## Local Optima Networks

A **Local Optima Network (LON)** is a directed graph where:

- **Nodes** represent local optima
- **Edges** represent transitions between optima discovered during search
- **Edge weights** indicate how often a transition was observed

### LON Construction

lonpy constructs LONs by:

1. Running multiple Basin-Hopping searches
2. Recording every accepted transition, where only non-worsening moves are accepted, so all LON edges are improving or equal (source optimum → target optimum)
3. Aggregating transitions into a weighted graph

```python
# Each transition creates an edge
# (optimum_A, fitness_A) → (optimum_B, fitness_B)
```

### Node Identification

Two solutions are considered the same local optimum if their coordinates match after rounding to `coordinate_precision` decimal places:

```python
# With coordinate_precision=5 (default):
# x1 = [1.234561, 2.345671] → "1.23456_2.34567"
# x2 = [1.234564, 2.345674] → "1.23456_2.34567"
# Same node!

# With coordinate_precision=None (full double precision):
# No rounding — only exact matches are the same node
```

## LON Metrics

lonpy computes several metrics to characterize fitness landscapes:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `n_optima` | Number of local optima | Problem multimodality |
| `n_funnels` | Number of sinks (no outgoing edges) | Distinct attraction basins |
| `n_global_funnels` | Sinks at global optimum | How many paths lead to success |
| `neutral` | Proportion of equal-fitness connections | Landscape flatness |
| `global_strength` | Incoming flow to global optima relative to all nodes | Global optimum accessibility |
| `sink_strength` | Incoming flow to global sinks relative to all sinks | Global sink dominance |
| `success` | Proportion of runs reaching global optimum | Search algorithm effectiveness |
| `deviation` | Mean absolute deviation from global optimum | Solution quality across runs |

### Interpreting Metrics

**Easy landscape** (single funnel):

- Few funnels (ideally 1)
- High global_strength and sink_strength (most flow reaches global)
- All paths converge to global optimum

**Hard landscape** (multiple funnels):

- Many funnels competing for flow
- Low global_strength and sink_strength (flow diverted to local sinks)
- Search easily gets trapped

## CMLON (Compressed Monotonic LON)

The **Compressed Monotonic LON (CMLON)** simplifies the network by merging nodes with equal fitness that are connected via equal-fitness edges.

This reveals the "downhill" structure of the landscape:

```
LON:  A ←→ B → C → D → E
      (equal)  (improving)

CMLON: [A,B] → C → D → E
       (merged)
```

### CMLON Colors

In CMLON visualizations:

- **Red**: Global optimum (best sink)
- **Blue**: Local sink (suboptimal endpoint)
- **Pink**: In global funnel (can reach global optimum)
- **Light blue**: In local funnel (trapped in suboptimal basin)

## Funnels

A **funnel** is a region of the fitness landscape where all descent paths converge to the same sink:

```
      A   B   C
       \ | /
        \|/
         D
         |
         E  ← sink
```

Funnels are identified as **sinks** in the LON or CMLON — nodes with no outgoing edges. Each sink represents an endpoint that search trajectories converge to.

### Global vs Local Funnels

- **Global funnel**: Leads to the global optimum
- **Local funnel**: Leads to a suboptimal local minimum

The ideal landscape has a single global funnel. Multiple funnels indicate potential difficulty for optimization algorithms.

## Further Reading

- [Sampling Guide](../user-guide/sampling.md) - Configure Basin-Hopping for your problem
- [Analysis Guide](../user-guide/analysis.md) - Interpret LON metrics
- [Visualization Guide](../user-guide/visualization.md) - Create plots of your LON
