import pytest

from lonpy import BasinHoppingSampler
from tests.conftest import DEFAULT_CONFIG, DOMAIN_2D, griewank, rastrigin, sphere


def count_evaluations_wrapper(func):
    count = 0

    def wrapper(*args):
        nonlocal count
        count += 1
        return func(*args)

    wrapper.nfev = lambda: count
    return wrapper


class TestNfevCounting:
    @pytest.mark.parametrize("test_func", [rastrigin, sphere, griewank])
    def test_nfev_counting(self, test_func) -> None:
        func = count_evaluations_wrapper(test_func)
        sampler = BasinHoppingSampler(DEFAULT_CONFIG)
        result = sampler.sample(func, DOMAIN_2D)
        assert result.nfev == func.nfev()
