import numpy as np
import pytest

from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig


class TestHashSolution:
    @pytest.mark.parametrize("precision", [0, 1, 2, 3])
    def test_small_positive_and_negative_values_hash_equally(self, precision: int) -> None:
        """Small values that round to zero should produce the same hash regardless of sign."""
        config = BasinHoppingSamplerConfig(coordinate_precision=precision)
        sampler = BasinHoppingSampler(config)

        threshold = 0.5 * 10 ** (-precision)
        small_pos = np.array([threshold / 2, threshold / 3])
        small_neg = -small_pos
        small_pos_rounded = sampler._round_value(small_pos, precision)
        small_neg_rounded = sampler._round_value(small_neg, precision)
        assert sampler._hash_solution(small_pos_rounded) == sampler._hash_solution(
            small_neg_rounded
        )

    def test_negative_zero_hashes_same_as_positive_zero(self) -> None:
        config = BasinHoppingSamplerConfig(coordinate_precision=1)
        sampler = BasinHoppingSampler(config)
        small_pos_rounded = sampler._round_value(np.array([0.0, 0.0]), 1)
        small_neg_rounded = sampler._round_value(np.array([-0.0, -0.0]), 1)
        assert sampler._hash_solution(small_pos_rounded) == sampler._hash_solution(
            small_neg_rounded
        )

    def test_negative_zero_hashes_same_as_positive_zero_no_precision(self) -> None:
        config = BasinHoppingSamplerConfig(coordinate_precision=None)
        sampler = BasinHoppingSampler(config)

        assert sampler._hash_solution(np.array([-0.0])) == sampler._hash_solution(np.array([0.0]))
