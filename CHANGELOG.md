# Changelog

## 0.3.0

Third public release adding support for discrete optimization problems via Iterated Local Search (ILS).

### Highlights

- Added a discrete optimization framework with `DiscreteProblem` abstract base class — a generic, stateless interface for defining custom discrete problems.
- Added `BitstringProblem` base class providing out-of-the-box `random_solution()`, `local_search()`, `perturb()`, and `solution_id()` for binary-encoded problems, with configurable first-improvement (stochastic) and best-improvement (deterministic) hill climbing.
- Added built-in problem implementations:
  - `NumberPartitioning`: Number Partitioning Problem with configurable hardness parameter `k`, random instance generation via `instance_seed`, or explicit weights.
  - `OneMax`: Simple maximization benchmark with O(1) delta evaluation.
- Added `ILSSampler` and `ILSSamplerConfig` for constructing LONs from discrete problems via Iterated Local Search, with configurable stopping criteria (`n_iter_no_change`, `max_iter`) and equal-acceptance moves.
- Discrete and continuous sampling produce the same trace format (`[run, fit1, node1, fit2, node2]`), so `LON.from_trace_data()`, `CMLON`, metrics, and visualization all work unchanged.

### API and Behavior Changes

- Package now exports `DiscreteProblem`, `BitstringProblem`, `NumberPartitioning`, `OneMax`, `ILSSampler`, `ILSSamplerConfig`, and `ILSResult`.
- Internal module structure reorganized: continuous sampling moved to `lonkit.continuous.sampling`, discrete modules under `lonkit.discrete.problems` and `lonkit.discrete.sampling`.

### Documentation

- Updated user guide and API docs to cover the discrete framework, ILS sampling, and built-in problems.
- Added discrete quick-start example to README.

## 0.2.0

Second public release adding multiprocessing to Basin-Hopping sampling procedure.

### Highlights

- Added configurable parallel Basin-Hopping sampling execution via `joblib` with the new `n_jobs` configuration parameter.
- Added optional progress reporting with `verbose=True` (powered by `tqdm`) in both `sample(...)` and `compute_lon(...)`.
- The solution preserves reproducibility across sequential and parallel runs when `seed` is set.
- Added dedicated parallel reproducibility tests (`tests/test_parallel_sampling.py`).

### API and Behavior Changes

- `BasinHoppingSampler.sample(...)` now emphasizes returning a `BasinHoppingResult` object (`trace_df`, `raw_records`, `nfev`) in examples and docs.
- Internal sampling flow was refactored into explicit single-run, sequential, and parallel execution paths.

### Documentation

- Updated user guide pages for sampling and examples to cover `n_jobs`, reproducibility guarantees, and `verbose` progress display.
- Added API documentation page for the step size module and linked it from the API index.

### Dependencies

- Added required runtime dependencies: `joblib>=1.3.0` and `tqdm>=4.67.3`.

## 0.1.0

Initial public release of `lonkit`, a Python library for constructing, analyzing, and visualizing Local Optima Networks (LONs) for continuous optimization problems.

### Highlights

- Added end-to-end LON construction from objective functions via `compute_lon`.
- Added configurable Basin-Hopping sampling with support for:
  - stopping by `n_iter_no_change` and/or `max_iter`,
  - percentage and fixed perturbation modes,
  - bounded search domains,
  - custom `scipy.optimize.minimize` methods and options,
  - optional user-supplied initial points.
- Added `LON` and `CMLON` graph models built on `igraph` (Python bindings for igraph).
- Added landscape analysis metrics.
- Added `StepSizeEstimator` for estimating Basin-Hopping step sizes from a target escape rate.
- Added visualization tools for:
  - 2D network plots,
  - 3D landscape-style plots,
  - animated rotation GIFs,
  - batch generation of standard outputs for both LON and CMLON views.

### Examples

- Added worked examples and research-oriented example scripts under `examples/bioma`.
