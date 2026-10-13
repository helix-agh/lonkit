from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lonkit import LON, ILSSampler, ILSSamplerConfig, NumberPartitioning, OneMax


def edges_with_counts(lon: LON) -> set[tuple[str, str, int]]:
    names = lon.vertex_names
    return {(names[e.source], names[e.target], e["Count"]) for e in lon.graph.es}


class NaNForLeadingZero(OneMax):
    """OneMax whose local search returns NaN when its input starts with 0."""

    def local_search(self, solution, rng):
        optimum, fitness = super().local_search(solution, rng)
        return optimum, float("nan") if solution[0] == 0 else fitness


class TestRunsWithoutAcceptedTransitions:
    @pytest.mark.parametrize("preserve", [True, False])
    def test_run_is_recorded_as_self_loop(self, preserve: bool) -> None:
        """Runs that find the optimum in the initial local search keep it (issue #39)."""
        sampler = ILSSampler(
            ILSSamplerConfig(
                n_runs=3,
                max_iter=2,
                accept_equal=False,
                seed=42,
                preserve_runs_without_transitions=preserve,
            )
        )
        result = sampler.sample(OneMax(4))
        if not preserve:
            assert len(result.raw_records) == 6
            assert result.trace_df.empty
            lon = sampler.sample_to_lon(result)
            assert lon.n_vertices == 0
            assert lon.final_run_values is None
            return
        assert result.trace_df.to_dict("records") == [
            {"run": run, "fit1": 4, "node1": "1111", "fit2": 4, "node2": "1111"}
            for run in (1, 2, 3)
        ]

        lon = sampler.sample_to_lon(result)
        assert lon.vertex_names == ["1111"]
        assert lon.n_edges == 0
        assert lon.best_fitness == 4
        assert lon.compute_performance_metrics(known_best=4) == {"success": 1.0, "deviation": 0.0}

    def test_mixed_runs_add_no_edges(self) -> None:
        problem = NumberPartitioning(n=10, weights=[3, 7, 11, 5, 13, 19, 2, 17, 23, 29])
        sampler = ILSSampler(ILSSamplerConfig(n_runs=20, max_iter=3, accept_equal=False, seed=0))
        result = sampler.sample(problem)
        trace = result.trace_df
        moving_runs = {r["run"] for r in result.raw_records if r["accepted"]}
        idle = trace[~trace["run"].isin(moving_runs)]
        assert 0 < len(idle) == trace["run"].nunique() - len(moving_runs)
        assert (idle["node1"] == idle["node2"]).all()

        lon = sampler.sample_to_lon(result)
        baseline = LON.from_trace_data(trace[trace["run"].isin(moving_runs)])
        assert set(lon.vertex_names) == set(baseline.vertex_names) | set(idle["node1"])
        assert edges_with_counts(lon) == edges_with_counts(baseline)
        assert lon.final_run_values is not None
        assert len(lon.final_run_values) == 20

        legacy_sampler = ILSSampler(
            replace(sampler.config, preserve_runs_without_transitions=False)
        )
        legacy = legacy_sampler.sample(problem)
        assert legacy.raw_records == result.raw_records
        pd.testing.assert_frame_equal(
            legacy.trace_df, trace[trace["run"].isin(moving_runs)].reset_index(drop=True)
        )
        legacy_lon = legacy_sampler.sample_to_lon(legacy)
        assert edges_with_counts(legacy_lon) == edges_with_counts(lon)
        assert legacy_lon.final_run_values is not None
        assert set(legacy_lon.final_run_values.index) == moving_runs

    def test_non_finite_initial_fitness_is_skipped(self) -> None:
        sampler = ILSSampler(ILSSamplerConfig(n_runs=10, max_iter=2, accept_equal=False, seed=0))
        result = sampler.sample(NaNForLeadingZero(4))
        initial = {r["run"]: r["current_fitness"] for r in result.raw_records}
        finite_runs = {run for run, fit in initial.items() if np.isfinite(fit)}
        assert 0 < len(finite_runs) < 10
        assert set(result.trace_df["run"]) == finite_runs
