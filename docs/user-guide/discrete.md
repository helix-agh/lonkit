# Discrete Problems

lonkit supports Local Optima Networks for discrete optimization problems using
Iterated Local Search (ILS).

## Concepts

### How Discrete LONs Differ from Continuous LONs

| Aspect | Continuous | Discrete |
|--------|-----------|----------|
| **Sampling method** | Basin-Hopping (perturbation + local minimizer) | Iterated Local Search (perturbation + hill climbing) |
| **Local search** | Scipy minimizer (L-BFGS-B, etc.) | Problem-specific neighborhood search (e.g., bit-flip) |
| **Perturbation** | Random displacement in continuous space | Combinatorial move (e.g., flip k bits) |
| **Node identity** | Rounded coordinates (hash) | Exact solution representation (e.g., bitstring) |
| **Neighborhoods** | Implicit (minimizer handles it) | Explicit (defined by the problem) |

Despite these differences, the downstream pipeline is identical: both produce
trace data in the format `[run, fit1, node1, fit2, node2]`, which feeds into
`LON.from_trace_data()` → `LON.to_cmlon()` → `LONVisualizer`.

## Built-in Problems

### NumberPartitioning

The **Number Partitioning Problem (NPP)** asks: given a set of N positive integers,
partition them into two subsets such that the absolute difference of their sums is
minimized.

```python
from lonkit import NumberPartitioning

# Random instance with hardness parameter k
problem = NumberPartitioning(n=20, k=0.5, instance_seed=1)

# Or provide explicit weights
problem = NumberPartitioning(n=4, weights=[7, 5, 6, 4])
```

**Parameters:**

- `n`: Number of items
- `k`: Hardness parameter. Items drawn uniformly from `[1, 2^(n*k)]`. Higher k produces harder instances.
- `instance_seed`: Seed for generating item weights (reproducible instances)
- `weights`: Explicit item weights (alternative to k + instance_seed)
- `n_perturbation_flips`: Number of random bit flips per perturbation (default: 2)
- `first_improvement`: Use first-improvement hill climbing (default: True)

**Fitness:** `|sum(A) - sum(B)|` (minimization, optimal = 0)

### OneMax

**OneMax** is a simple benchmark: maximize the number of 1-bits in a bitstring.

```python
from lonkit import OneMax

problem = OneMax(n=20)
```

**Parameters:**

- `n`: Length of the bitstring
- `n_perturbation_flips`: Number of random bit flips per perturbation (default: 2)
- `first_improvement`: Use first-improvement hill climbing (default: True)

**Fitness:** `sum(bits)` (maximization, optimal = n)

OneMax has a single global optimum (all ones) and supports O(1) delta evaluation.

## Custom Problems

To define your own discrete problem, you can subclass `DiscreteProblem`:

```python
import random
from lonkit import DiscreteProblem

class MaxCut(DiscreteProblem[list[int]]):
    """Maximize the number of edges crossing the partition."""

    def __init__(self, adjacency: list[list[int]]):
        self.adjacency = adjacency
        self.n = len(adjacency)

    @property
    def minimize(self) -> bool:
        return False  # maximization

    def random_solution(self, rng: random.Random) -> list[int]:
        return [rng.randint(0, 1) for _ in range(self.n)]

    def evaluate(self, solution: list[int]) -> float:
        cut = 0
        for i in range(self.n):
            for j in self.adjacency[i]:
                if solution[i] != solution[j]:
                    cut += 1
        return float(cut // 2)  # each edge counted twice

    def local_search(self, solution: list[int], rng: random.Random) -> tuple[list[int], float]:
        sol = list(solution)
        fitness = self.evaluate(sol)
        improved = True
        while improved:
            improved = False
            indices = list(range(self.n))
            rng.shuffle(indices)
            for i in indices:
                sol[i] = 1 - sol[i]
                new_fitness = self.evaluate(sol)
                if self.is_better(new_fitness, fitness):
                    fitness = new_fitness
                    improved = True
                    break
                sol[i] = 1 - sol[i]
        return sol, fitness

    def perturb(self, solution: list[int], rng: random.Random) -> list[int]:
        sol = list(solution)
        indices = rng.sample(range(self.n), min(3, self.n))
        for i in indices:
            sol[i] = 1 - sol[i]
        return sol

    def solution_id(self, solution: list[int]) -> str:
        return "".join(str(b) for b in solution)
```

### Abstract Methods

| Method | Purpose |
|--------|---------|
| `minimize` (property) | Return `True` for minimization, `False` for maximization |
| `random_solution(rng)` | Generate a random initial solution |
| `evaluate(solution)` | Return the fitness value (pure function, no side effects) |
| `local_search(solution, rng)` | Run hill climbing to a local optimum; return `(solution, fitness)` |
| `perturb(solution, rng)` | Apply perturbation to escape current basin |
| `solution_id(solution)` | Return a unique string identifier for the solution |

### Using BitstringProblem

For bitstring-based problems, inherit from `BitstringProblem` instead. It provides
`random_solution()`, `local_search()`, `perturb()`, and `solution_id()` out of the box.
You only need to implement `evaluate()`:

```python
from lonkit import BitstringProblem

class LeadingOnes(BitstringProblem):
    """Count consecutive leading 1-bits."""

    @property
    def minimize(self) -> bool:
        return False

    def evaluate(self, solution: list[int]) -> float:
        count = 0
        for bit in solution:
            if bit == 1:
                count += 1
            else:
                break
        return float(count)
```

**Tips:**

- Override `delta_evaluate(solution, index)` for O(1) neighbor evaluation when possible
- Set `first_improvement=False` for deterministic best-improvement hill climbing
- Adjust `n_perturbation_flips` to control perturbation strength

**First-improvement vs best-improvement:**

- **First-improvement** (`first_improvement=True`, default): The scan order over neighbor
  positions is randomized each pass. As soon as an improving neighbor is found, the move
  is accepted immediately. This makes the local search *stochastic* — the same starting
  solution may reach different local optima depending on the RNG state.
- **Best-improvement** (`first_improvement=False`): All neighbors are evaluated every pass
  and the best improving move is selected. The scan order does not affect the outcome,
  making the local search *deterministic* (given the same starting solution, it always
  reaches the same local optimum).

## ILS Configuration

Configure the sampling process with `ILSSamplerConfig`:

```python
from lonkit import ILSSamplerConfig

config = ILSSamplerConfig(
    n_runs=100,              # Number of independent ILS runs
    n_iter_no_change=100,    # Stop after 100 non-improving iterations
    max_iter=None,           # No hard iteration limit (use n_iter_no_change)
    accept_equal=True,       # Accept lateral moves (equal fitness)
    seed=42,                 # For reproducibility
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | 100 | Number of independent ILS restarts. More runs = better coverage of the landscape |
| `n_iter_no_change` | 100 | Patience: stop a run after this many consecutive non-improving iterations. Set to `None` to disable (must set `max_iter`) |
| `max_iter` | None | Hard limit on total iterations per run. Set to `None` for no limit (must set `n_iter_no_change`) |
| `accept_equal` | True | If `True`, accept moves to equal-fitness optima (greedy with equal acceptance). If `False`, only strictly improving moves |
| `seed` | None | Random seed controlling all search randomness |

At least one of `n_iter_no_change` or `max_iter` must be set.

### Comparison with BasinHoppingSamplerConfig

| ILS Parameter | Basin-Hopping Equivalent | Notes |
|---------------|-------------------------|-------|
| `n_runs` | `n_runs` | Same meaning |
| `n_iter_no_change` | `n_iter_no_change` | Same stopping criterion |
| `accept_equal` | — | Discrete-specific; continuous uses tolerance-based comparison |
| `seed` | `seed` | Same meaning |

## Complete Example

```python
from lonkit import (
    NumberPartitioning,
    ILSSampler,
    ILSSamplerConfig,
    LONVisualizer,
)

# Create problem instance
problem = NumberPartitioning(n=20, k=0.5, instance_seed=1)

# Configure sampler
config = ILSSamplerConfig(
    n_runs=50,
    n_iter_no_change=100,
    seed=42,
)
sampler = ILSSampler(config)

# Run sampling
result = sampler.sample(problem)

# Build LON
lon = sampler.sample_to_lon(result)
print(f"LON: {lon.n_vertices} optima, {lon.n_edges} edges")

# Build CMLON
cmlon = lon.to_cmlon()
print(f"CMLON: {cmlon.n_vertices} optima, {cmlon.n_edges} edges")

# Compute metrics
metrics = cmlon.compute_metrics()
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(cmlon, output_path="npp_cmlon_2d.png")
viz.plot_3d(cmlon, output_path="npp_cmlon_3d.png")
```

## API Reference

See the [Discrete API Reference](../api/discrete.md) for complete API documentation.
