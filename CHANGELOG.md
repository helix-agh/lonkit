# Changelog

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
- Added `LON` and `CMLON` graph models built on `python-igraph`.
- Added landscape analysis metrics.
- Added `StepSizeEstimator` for estimating Basin-Hopping step sizes from a target escape rate.
- Added visualization tools for:
  - 2D network plots,
  - 3D landscape-style plots,
  - animated rotation GIFs,
  - batch generation of standard outputs for both LON and CMLON views.

### Examples

- Added worked examples and research-oriented example scripts under `examples/bioma`.
