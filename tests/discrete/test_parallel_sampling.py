import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from lonkit import ILSSampler, ILSSamplerConfig, NKLandscape, NumberPartitioning


@pytest.mark.parametrize(
    "problem",
    [
        NumberPartitioning(n=10, weights=[3, 7, 11, 5, 13, 19, 2, 17, 23, 29]),
        NKLandscape(n=10, k=3, instance_seed=7),
    ],
    ids=["minimize", "maximize"],
)
def test_parallel_results_match_sequential(problem):
    config = ILSSamplerConfig(n_runs=6, max_iter=12, n_iter_no_change=5, seed=42)
    sequential = ILSSampler(config).sample(problem)
    parallel = ILSSampler(replace(config, n_jobs=2)).sample(problem)

    # These landscapes expose RNG differences, unlike a single-optimum OneMax case.
    assert len({record["new_id"] for record in sequential.raw_records}) > 1
    assert sequential.raw_records == parallel.raw_records
    pd.testing.assert_frame_equal(sequential.trace_df, parallel.trace_df, check_exact=True)


class CompletionRecordingSampler(ILSSampler):
    """Record completion at the run boundary, including the executing process."""

    def __init__(self, config, completion_dir):
        super().__init__(config)
        self.completion_dir = completion_dir

    def _ils_run(self, problem, run, rng):
        records = super()._ils_run(problem, run, rng)
        (self.completion_dir / str(run)).write_text(str(os.getpid()))
        return records


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_progress_reports_completed_runs_in_main_process(n_jobs, tmp_path):
    main_pid = os.getpid()
    calls = []
    config = ILSSamplerConfig(n_runs=4, max_iter=3, seed=42, n_jobs=n_jobs)
    sampler = CompletionRecordingSampler(config, tmp_path)

    def progress(completed, total):
        assert os.getpid() == main_pid
        # A notification must follow actual completion, including in sequential mode.
        assert len(list(tmp_path.iterdir())) >= completed
        calls.append((completed, total))

    sampler.sample(NumberPartitioning(n=8, k=0.5, instance_seed=1), progress)
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    run_pids = {int(path.read_text()) for path in Path(tmp_path).iterdir()}
    if n_jobs == 1:
        assert run_pids == {main_pid}
    else:
        assert main_pid not in run_pids


class FailingProblem(NumberPartitioning):
    def random_solution(self, rng):
        raise RuntimeError(f"Search failed in process {os.getpid()}")


def test_worker_exception_is_propagated():
    sampler = ILSSampler(ILSSamplerConfig(n_runs=2, max_iter=3, seed=42, n_jobs=2))
    with pytest.raises(RuntimeError, match="Search failed in process") as error:
        sampler.sample(FailingProblem(n=4, weights=[1, 2, 3, 4]))
    assert str(error.value) != f"Search failed in process {os.getpid()}"


def test_zero_n_jobs_is_rejected():
    with pytest.raises(ValueError, match="n_jobs"):
        ILSSamplerConfig(max_iter=3, n_jobs=0)
