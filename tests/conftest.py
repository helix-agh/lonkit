import numpy as np
import pytest

from lonpy import LON, BasinHoppingSampler, BasinHoppingSamplerConfig

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


@pytest.fixture(scope="session")
def sphere_lon() -> LON:
    sampler = BasinHoppingSampler(DEFAULT_CONFIG)
    return sampler.sample_to_lon(sphere, DOMAIN_2D)


@pytest.fixture(scope="session")
def rastrigin_lon() -> LON:
    sampler = BasinHoppingSampler(DEFAULT_CONFIG)
    return sampler.sample_to_lon(rastrigin, DOMAIN_2D)
