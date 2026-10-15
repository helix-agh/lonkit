# Changelog

## Unreleased

### Bug Fixes

- `LONVisualizer.compute_lon_colors()` and `compute_cmlon_colors()`, and therefore all plots, now classify global optima using the network's `eq_atol`, like the network metrics, instead of exact equality. `CMLON.get_local_sinks()` now returns exactly the sinks that are not global sinks, so a sink within `eq_atol` of the best fitness is no longer both a global and a local sink. Previously, such sinks counted towards `n_global_funnels` but were drawn as local sinks. Plots of networks with rounded fitness (`fitness_precision`) and the default `eq_atol` are unchanged.
- Raised the minimum Plotly version to 6.1.1 and Kaleido to 1.0.0 to exclude incompatible image-export dependencies. Documented the Chrome/Chromium requirement and added real PNG and rotation GIF export tests, with Chrome installation in CI. `plot_3d()` now passes image dimensions explicitly to the exporter so the minimum supported versions preserve the requested size.
- `LONVisualizer.get_layout()` with a `seed`, and therefore `plot_2d()`, `plot_3d()`, `create_rotation_gif()` and `visualize_all()`, no longer reseeds the global NumPy RNG or leaves a seeded RNG installed in igraph. The seed now applies only to the layout, and igraph's default RNG is restored afterwards. Seeded layouts are unchanged (#53).
- Continuous sampling now handles failed local searches explicitly. A local search fails when `minimize` raises `ValueError` or returns a non-finite fitness. A failed initial local search fails its run, and a failed search after a perturbation is rejected and counts as a non-improving iteration. Previously, runs starting at a `nan` minimum silently vanished, and an `inf` minimum made `LON.from_trace_data()` fail after all sampling had finished. Results with `success=False` and a finite fitness are still used as local optima. Each failed local search emits a warning with its context and reason (#44).
- `StepSizeEstimator` divides each sample's escape count by its number of evaluated perturbations instead of `n_perturbations`, so failed local searches no longer bias the escape rate down. Samples whose initial local search fails are skipped, and `RuntimeError` is raised if no sample can be evaluated, instead of returning `nan` (#44).
- Runs without accepted transitions are no longer dropped from the LON and its performance metrics. `BasinHoppingSampler` and `ILSSampler` now record such a run as a single trace row from its initial local optimum to itself. `LON.from_trace_data()` drops self-loops, so the optimum becomes a vertex without edges and the run counts towards `success` and `deviation`. Basin-Hopping runs with a non-finite initial minimum are still skipped. Seeded traces and LONs change only when some run had no accepted transitions. Set `preserve_runs_without_transitions=False` in either sampler config to restore the earlier behavior, excluding these runs from both the graph and performance metrics (#39).

- Raised the minimum joblib version to 1.4.0, which introduced the `generator_unordered` mode required by parallel Basin-Hopping sampling.
- `BitstringProblem.local_search()` now returns a full `evaluate()` of the final solution instead of the fitness accumulated from deltas. For `NKLandscape`, float drift gave the same local optimum slightly different fitness values on different visits. ILS then counted returns to the same optimum as improvements, and the LON had duplicate nodes with multiple fitness values. Seeded NK samples may differ from earlier versions.

## 0.8.0

Adds parallel Iterated Local Search sampling for discrete problems.

### Highlights

- Added `ILSSamplerConfig.n_jobs` for parallel ILS runs through joblib, preferring processes. Effective `n_jobs=1` executes directly without workers.
- ILS now spawns a separate RNG per run, producing identical ordered records and traces across job counts for a fixed seed. Seeded results differ from earlier versions that shared one RNG across runs.
- ILS progress callbacks now run in the main process after each completed run is received, reporting completed and total run counts. Results are received in run order. Worker errors propagate to the caller.
- Documented parallel ILS usage, custom problem serialization, and process overhead for short runs.

### Bug Fixes

- Parallel Basin-Hopping progress callbacks now report the number of completed runs instead of the ID of the run that just finished, so progress increases monotonically even when runs finish out of order.

### Examples

- BIOMA figure titles now use `D` for problem dimension.

## 0.7.0

Adds the number of edges to the network metrics.

### Highlights

- `LON.compute_network_metrics()` and `CMLON.compute_network_metrics()`, and therefore `compute_metrics()`, now also return `n_edges`, the number of edges in the network.

## 0.6.0

Validates trace data in `LON.from_trace_data()`, making it safe to build LONs from traces produced outside of lonkit, e.g. by samplers implemented in other languages, and makes LON construction much faster for large traces.

### Highlights

- `LON.from_trace_data()` now validates the trace before building the LON:
  - raises `ValueError` for malformed data: an empty trace, missing or extra columns, missing values, non-integer run numbers, non-numeric (including datetime and boolean) or infinite fitness values, and non-string node identifiers;
  - emits a `UserWarning` for broken trajectories, i.e. when within a run `node2` of a row differs from `node1` of the next row, which usually means that rows are out of order or include rejected moves.
- `LON.from_trace_data()` builds the graph in bulk instead of adding vertices and edges one by one, so construction time now grows linearly instead of quadratically with the trace size (e.g. ~8× faster for 1,000 trace rows and ~590× faster for 50,000).

### API and Behavior Changes

- Trace columns are matched by name and must be exactly `run`, `fit1`, `node1`, `fit2`, `node2`, in any order. Previously, columns were taken by position regardless of their names, so DataFrames with other column names must now be renamed.
- An empty trace now raises `ValueError` instead of returning a LON without vertices and with a `NaN` best fitness. `sample_to_lon()` still returns an empty `LON` for an empty sampling result.
- The documented column types are now enforced: `run` must be an integer column and node identifiers must be strings. In particular, a CSV file read without `dtype={"node1": str, "node2": str}` is rejected instead of silently losing leading zeros of identifiers such as bitstrings.

### Bug Fixes

- Non-string node identifiers now raise `ValueError` instead of silently dropping all edges. For integer identifiers, vertices were named after a float-converted identifier (`"10.0"`) while edges were looked up by the original one (`"10"`), and the resulting errors were ignored. The built-in samplers always produce string identifiers and were not affected.

### Documentation

- Added a "Loading External Traces" section to the sampling user guide.

## 0.5.0

Fixes CMLON construction so that the resulting graph is always monotonic, regardless of the input LON.

### Bug Fixes

- `CMLON.from_lon()` now removes worsening edges instead of keeping them. Previously, monotonicity held only if the input LON was monotonic, which could fail in two ways:
  - user-supplied traces passed to `LON.from_trace_data()` containing non-improving transitions (e.g. from a sampler with a non-elitist acceptance rule);
  - node deduplication: the same node recorded with slightly different fitness values receives a single aggregated value (`fitness_aggregation`), which can lie below the fitness of its successor. This could happen with the built-in samplers under default settings (`fitness_precision=None`).
- Neutral-component compression is now repeated until no equal-fitness edges remain, because approximate equality (within `eq_atol`) is not transitive: a contracted component can become equal to a neighbour it was not equal to before contraction.
- After compression, edges are checked again against the representative fitness of each component; edges that became worsening are removed.
- Compressed components retain their best fitness (`min` for minimization, `max` for maximization), preserving the global optimum through repeated compression.
- Component names come from the vertex with the retained fitness; ties use the first vertex in graph order.

### API and Behavior Changes

- `CMLON.from_lon()` / `LON.to_cmlon()` emit a `UserWarning` whenever worsening edges are removed (both before and after compression).
- For non-monotonic input LONs, CMLON metrics (`n_funnels`, `n_global_funnels`, `sink_strength`, `global_funnel_proportion`) may differ from 0.4.0, since nodes left only via worsening edges are now sinks.
- `LON.from_trace_data()` still keeps edges exactly as recorded, so a `LON` itself is not required to be monotonic.

### Documentation

- Documented the handling of worsening edges in `LON.from_trace_data()`, `CMLON.from_lon()` and the concepts guide.

### Tests

- Added `tests/test_cmlon_monotonicity.py` covering non-elitist traces (minimization and maximization), deduplication with every fitness aggregation strategy, non-transitive neutral chains, randomized invariant checks, and a Basin-Hopping integration test on Schwefel 2.26.

## 0.4.0

Adds Kauffman's NK Landscape as a built-in discrete benchmark problem.

### Highlights

- Added `NKLandscape`, a tunable family of rugged fitness landscapes over bitstrings:
  - `k` controls ruggedness, from smooth (`k=0`) to maximally rugged (`k=n-1`).
  - `neighbor_model` selects `"adjacent"` (cyclic) or `"random"` distinct epistatic neighbors.
  - Fixed instance (neighbor structure and contribution tables) generated from `instance_seed` for reproducibility.
  - Vectorized `evaluate()` and O(affected-positions) `delta_evaluate()` for fast hill climbing.
- This is a maximization problem (optimal fitness close to 1.0).

### API and Behavior Changes

- Package now exports `NKLandscape`.

### Documentation

- Added `NKLandscape` to the user guide and API reference.

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
