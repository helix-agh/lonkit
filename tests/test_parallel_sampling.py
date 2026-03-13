"""
Tests for the multiprocessing Basin-Hopping implementation.

The central guarantee being tested: results produced by ``n_jobs=1``
(sequential) are bit-for-bit identical to those produced by any ``n_jobs>1``
(parallel) value, as long as ``seed`` is fixed.  This property holds because
every run derives its RNG from a deterministic ``SeedSequence`` tree, so the
per-run random state is independent of how many parallel workers are used.
"""

import os
from functools import partial

import numpy as np
import pandas as pd
import pytest

from lonkit import BasinHoppingSampler, BasinHoppingSamplerConfig

# ---------------------------------------------------------------------------
# Module-level helper functions
# All must be at module scope so that joblib's loky backend can pickle them.
# ---------------------------------------------------------------------------

DOMAIN_2D = [(-5.0, 5.0), (-5.0, 5.0)]


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def _record_pid_and_eval(x: np.ndarray, func, pid_path: str) -> float:
    """Write the current process PID to *pid_path*, then evaluate *func*."""
    with open(pid_path, "a") as f:
        f.write(f"{os.getpid()}\n")
    return func(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(seed: int, n_jobs: int | None = 1, n_runs: int = 8, max_iter: int = 20):
    return BasinHoppingSamplerConfig(
        n_runs=n_runs,
        max_iter=max_iter,
        seed=seed,
        n_jobs=n_jobs,
    )


def _run(seed: int, n_jobs: int | None = 1, func=_sphere):
    sampler = BasinHoppingSampler(_make_config(seed=seed, n_jobs=n_jobs))
    return sampler.sample(func, DOMAIN_2D)


def _assert_raw_records_equal(seq_records: list[dict], par_records: list[dict]) -> None:
    """Assert that two raw-record lists are identical element by element."""
    assert len(seq_records) == len(par_records), (
        f"Record count differs: {len(seq_records)} (sequential) vs {len(par_records)} (parallel)"
    )
    for i, (s, p) in enumerate(zip(seq_records, par_records)):
        assert s["run"] == p["run"], f"Record {i}: run mismatch"
        assert s["iteration"] == p["iteration"], f"Record {i}: iteration mismatch"
        np.testing.assert_array_equal(
            s["current_x"], p["current_x"], err_msg=f"Record {i}: current_x mismatch"
        )
        assert s["current_f"] == p["current_f"], f"Record {i}: current_f mismatch"
        np.testing.assert_array_equal(s["new_x"], p["new_x"], err_msg=f"Record {i}: new_x mismatch")
        assert s["new_f"] == p["new_f"], f"Record {i}: new_f mismatch"
        assert s["accepted"] == p["accepted"], f"Record {i}: accepted mismatch"


# ---------------------------------------------------------------------------
# Reproducibility: parallel must match sequential bit-for-bit
# ---------------------------------------------------------------------------


class TestParallelReproducibility:
    """Parallel results must be identical to sequential results for the same seed."""

    SEED = 77

    def test_raw_records_match_sequential(self) -> None:
        """All perturbation records (accepted and rejected) are identical."""
        result_seq = _run(self.SEED, n_jobs=1)
        result_par = _run(self.SEED, n_jobs=2)
        _assert_raw_records_equal(result_seq.raw_records, result_par.raw_records)

    def test_nfev_matches_sequential(self) -> None:
        """Total function evaluation count is identical."""
        result_seq = _run(self.SEED, n_jobs=1)
        result_par = _run(self.SEED, n_jobs=2)
        assert result_seq.nfev == result_par.nfev

    def test_trace_df_matches_sequential(self) -> None:
        """The accepted-transitions DataFrame is identical."""
        result_seq = _run(self.SEED, n_jobs=1)
        result_par = _run(self.SEED, n_jobs=2)
        pd.testing.assert_frame_equal(result_seq.trace_df, result_par.trace_df)

    @pytest.mark.parametrize("n_jobs", [2, -1, None])
    def test_n_jobs_variants_match_sequential(self, n_jobs: int | None) -> None:
        """n_jobs=2, -1, and None all produce the same nfev and trace_df as n_jobs=1."""
        result_seq = _run(self.SEED, n_jobs=1)
        result_other = _run(self.SEED, n_jobs=n_jobs)
        assert result_seq.nfev == result_other.nfev
        pd.testing.assert_frame_equal(result_seq.trace_df, result_other.trace_df)

    @pytest.mark.parametrize("func", [_sphere, _rastrigin])
    def test_reproducibility_across_functions(self, func) -> None:
        """Reproducibility holds for different objective functions."""
        result_seq = _run(self.SEED, n_jobs=1, func=func)
        result_par = _run(self.SEED, n_jobs=2, func=func)
        _assert_raw_records_equal(result_seq.raw_records, result_par.raw_records)

    def test_different_seeds_give_different_results(self) -> None:
        """Sanity check: different seeds must produce different perturbation sequences."""
        result_a = _run(seed=1, n_jobs=2)
        result_b = _run(seed=2, n_jobs=2)
        # At least one record must differ (practically certain for any non-trivial run)
        first_f_a = [r["new_f"] for r in result_a.raw_records]
        first_f_b = [r["new_f"] for r in result_b.raw_records]
        assert first_f_a != first_f_b


# ---------------------------------------------------------------------------
# Progress callback behaviour
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Progress callback must behave consistently between sequential and parallel modes."""

    N_RUNS = 4
    SEED = 99

    def _run_with_callback(self, n_jobs: int | None) -> list[tuple[int, int]]:
        calls: list[tuple[int, int]] = []

        def callback(run: int, total: int) -> None:
            calls.append((run, total))

        config = _make_config(seed=self.SEED, n_jobs=n_jobs, n_runs=self.N_RUNS, max_iter=10)
        sampler = BasinHoppingSampler(config)
        sampler.sample(_sphere, DOMAIN_2D, progress_callback=callback)
        return calls

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_callback_called_n_runs_times(self, n_jobs: int) -> None:
        """Callback is invoked exactly n_runs times regardless of n_jobs."""
        calls = self._run_with_callback(n_jobs)
        assert len(calls) == self.N_RUNS

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_callback_total_argument_is_n_runs(self, n_jobs: int) -> None:
        """The second callback argument is always n_runs."""
        calls = self._run_with_callback(n_jobs)
        assert all(total == self.N_RUNS for _, total in calls)

    def test_callback_arguments_match_between_modes(self) -> None:
        """Sequential and parallel modes pass identical (run, total) pairs to the callback."""
        calls_seq = self._run_with_callback(n_jobs=1)
        calls_par = self._run_with_callback(n_jobs=2)
        assert calls_seq == calls_par

    def test_callback_called_in_main_process(self) -> None:
        """The callback is always invoked from the main process, never from a worker."""
        main_pid = os.getpid()
        callback_pids: list[int] = []

        def callback(run: int, total: int) -> None:
            callback_pids.append(os.getpid())

        config = _make_config(seed=self.SEED, n_jobs=2, n_runs=self.N_RUNS, max_iter=10)
        sampler = BasinHoppingSampler(config)
        sampler.sample(_sphere, DOMAIN_2D, progress_callback=callback)

        assert len(callback_pids) == self.N_RUNS
        assert all(pid == main_pid for pid in callback_pids), (
            "Callback was invoked from a worker process. "
            f"Main PID: {main_pid}, callback PIDs: {callback_pids}"
        )


# ---------------------------------------------------------------------------
# Multiprocessing backend: processes, not threads
# ---------------------------------------------------------------------------


class TestMultiprocessingBackend:
    """Verify that parallel execution spawns separate OS processes."""

    def test_parallel_uses_separate_processes(self, tmp_path: pytest.TempPathFactory) -> None:
        """
        Worker evaluations must run in a different OS process than the main
        process.  Each function call appends its PID to a shared file; we then
        verify that none of those PIDs belong to the main process.
        """
        pid_file = str(tmp_path / "worker_pids.txt")
        func = partial(_record_pid_and_eval, func=_sphere, pid_path=pid_file)

        config = BasinHoppingSamplerConfig(
            n_runs=4,
            max_iter=5,
            seed=42,
            n_jobs=2,
        )
        sampler = BasinHoppingSampler(config)
        sampler.sample(func, DOMAIN_2D)

        with open(pid_file) as f:
            worker_pids = {int(line.strip()) for line in f if line.strip()}

        assert len(worker_pids) > 0, "No PIDs were recorded by worker processes."
        assert os.getpid() not in worker_pids, (
            "The main process PID appeared in worker PIDs, indicating threads were used."
        )
