"""Regression coverage for CMLON reference optima (issue #41)."""

import pandas as pd
import pytest

from lonkit import LON, LONConfig


@pytest.mark.parametrize("method", ["compute_network_metrics", "compute_metrics"])
@pytest.mark.parametrize("minimize", [True, False])
@pytest.mark.parametrize(
    ("known_best", "expected_funnels", "expected_proportion"),
    [(0.0, 0, 0.0), (1.0, 1, 1.0), (None, 1, 1.0)],
)
def test_global_funnel_proportion_uses_known_best(
    method, minimize, known_best, expected_funnels, expected_proportion
):
    sign = 1 if minimize else -1
    trace = pd.DataFrame(
        [(1, sign * 2.0, "a", sign * 1.0, "b")],
        columns=["run", "fit1", "node1", "fit2", "node2"],
    )
    cmlon = LON.from_trace_data(trace, LONConfig(minimize=minimize)).to_cmlon()
    reference = sign * known_best if known_best is not None else None

    metrics = getattr(cmlon, method)(known_best=reference)

    assert metrics["n_global_funnels"] == expected_funnels
    assert metrics["global_funnel_proportion"] == expected_proportion
