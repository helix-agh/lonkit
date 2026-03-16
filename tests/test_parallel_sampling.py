import numpy as np
import pandas as pd
import pytest

from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig
from tests.conftest import DOMAIN_2D, SEED, griewank, rastrigin, sphere


def _make_config(seed: int, n_jobs: int | None = 1, n_runs: int = 8, max_iter: int = 20):
    return BasinHoppingSamplerConfig(
        n_runs=n_runs,
        max_iter=max_iter,
        seed=seed,
        n_jobs=n_jobs,
    )


def _run(seed: int, n_jobs: int | None = 1, func=sphere):
    sampler = BasinHoppingSampler(_make_config(seed, n_jobs))
    return sampler.sample(func, DOMAIN_2D)


def _assert_raw_records_equal(seq_records: list[dict], par_records: list[dict]) -> None:
    """Assert that two raw-record lists are identical element by element."""
    assert len(seq_records) == len(par_records), (
        f"Record count differs: {len(seq_records)} (sequential) vs {len(par_records)} (parallel)"
    )

    sorted_seq = sorted(seq_records, key=lambda x: x["run"])
    sorted_par = sorted(par_records, key=lambda x: x["run"])

    def _check_dict(d1: dict, d2: dict) -> bool:
        # Compare all keys except 'timestamp' (which may differ due to execution timing)
        assert d1["run"] == d2["run"], "Run numbers differ"
        assert d1["iteration"] == d2["iteration"], "Iteration numbers differ"
        assert d1["accepted"] == d2["accepted"], "Acceptance flags differ"

        assert np.array_equal(d1["current_x"], d2["current_x"]), "Current solution differs"
        assert np.array_equal(d1["current_f"], d2["current_f"]), "Current function value differs"

        assert np.array_equal(d1["new_x"], d2["new_x"]), "new solution differs"
        assert np.array_equal(d1["new_f"], d2["new_f"]), "new function value differs"

        return True

    for d1, d2 in zip(sorted_seq, sorted_par):
        _check_dict(d1, d2)


class TestParallelReproducibility:
    """Parallel results must be identical to sequential results for the same seed."""

    def test_raw_records_match_sequential(self) -> None:
        """All perturbation records (accepted and rejected) are identical."""
        result_seq = _run(SEED, n_jobs=1)
        result_par = _run(SEED, n_jobs=-1)
        _assert_raw_records_equal(result_seq.raw_records, result_par.raw_records)

    def test_nfev_matches_sequential(self) -> None:
        """Total function evaluation count is identical."""
        result_seq = _run(SEED, n_jobs=1)
        result_par = _run(SEED, n_jobs=-1)
        assert result_seq.nfev == result_par.nfev

    def test_trace_df_matches_sequential(self) -> None:
        """The accepted-transitions DataFrame is identical."""
        result_seq = _run(SEED, n_jobs=1)
        result_par = _run(SEED, n_jobs=-1)

        pd.testing.assert_frame_equal(result_seq.trace_df, result_par.trace_df)

    @pytest.mark.parametrize("n_jobs", [2, -1, None])
    def test_n_jobs_variants_match_sequential(self, n_jobs: int | None) -> None:
        """n_jobs=2, -1, and None all produce the same nfev and trace_df as n_jobs=1."""
        result_seq = _run(SEED, n_jobs=1)
        result_other = _run(SEED, n_jobs=n_jobs)
        assert result_seq.nfev == result_other.nfev

        pd.testing.assert_frame_equal(result_seq.trace_df, result_other.trace_df)

    @pytest.mark.parametrize("func", [sphere, rastrigin, griewank])
    def test_reproducibility_across_functions(self, func) -> None:
        """Reproducibility holds for different objective functions."""
        result_seq = _run(SEED, n_jobs=1, func=func)
        result_par = _run(SEED, n_jobs=2, func=func)
        _assert_raw_records_equal(result_seq.raw_records, result_par.raw_records)
