import numpy as np
import pytest

from lonkit import LON, BasinHoppingSampler, BasinHoppingSamplerConfig

SEED = 42
DOMAIN_2D = [(-5.0, 5.0), (-5.0, 5.0)]
DEFAULT_CONFIG = BasinHoppingSamplerConfig(
    n_runs=5,
    max_iter=50,
    seed=SEED,
)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2 / 4000.0) - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


@pytest.fixture(scope="session")
def sphere_lon() -> LON:
    sampler = BasinHoppingSampler(DEFAULT_CONFIG)
    return sampler.sample_to_lon(sampler.sample(sphere, DOMAIN_2D))


@pytest.fixture(scope="session")
def rastrigin_lon() -> LON:
    sampler = BasinHoppingSampler(DEFAULT_CONFIG)
    return sampler.sample_to_lon(sampler.sample(rastrigin, DOMAIN_2D))
