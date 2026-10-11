import pandas as pd
import pytest

from lonkit import LON


def test_self_loops_removed() -> None:
    trace = pd.DataFrame(
        {
            "run": [1, 1, 1],
            "fit1": [3.0, 2.0, 2.0],
            "node1": ["a", "b", "b"],
            "fit2": [2.0, 2.0, 1.0],
            "node2": ["b", "b", "c"],
        }
    )
    lon = LON.from_trace_data(trace)
    assert lon.graph.get_edgelist() == [(0, 1), (1, 2)]
    assert lon.vertex_count == [1, 4, 1]


def test_only_self_loops_gives_graph_without_edges() -> None:
    trace = pd.DataFrame({"run": [1], "fit1": [1.0], "node1": ["a"], "fit2": [1.0], "node2": ["a"]})
    lon = LON.from_trace_data(trace)
    assert lon.n_vertices == 1
    assert lon.n_edges == 0


def test_integer_node_ids_raise() -> None:
    trace = pd.DataFrame(
        {
            "run": [1, 1],
            "fit1": [3.0, 2.0],
            "node1": [10, 20],
            "fit2": [2.0, 1.0],
            "node2": [20, 30],
        }
    )
    with pytest.raises(ValueError, match="must contain string node identifiers"):
        LON.from_trace_data(trace)
