import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import igraph as ig
import numpy as np
import pandas as pd

DEFAULT_ATOL = 1e-12


@dataclass
class LONConfig:
    """
    Configuration for LON construction from trace data.

    Attributes:
        fitness_aggregation: Strategy for handling nodes with multiple fitness values:
            - `"min"`: Use minimum fitness
            - `"max"`: Use maximum fitness
            - `"mean"`: Use average fitness
            - `"first"`: Use first occurrence
            - `"strict"`: Raise error if duplicates are detected
            Default: `"min"`.
        warn_on_duplicates: Whether to emit a warning when duplicate nodes detected. Default: `True`
        max_fitness_deviation: If set, raise error if fitness deviation exceeds this threshold. Default: `None` (no threshold).
            Useful for detecting data quality issues.
        eq_atol: Tolerance for considering fitness values as equal. Default: `1e-12`.
    """

    fitness_aggregation: Literal["min", "max", "mean", "first", "strict"] = "min"
    warn_on_duplicates: bool = True
    max_fitness_deviation: float | None = None
    eq_atol: float = DEFAULT_ATOL


@dataclass
class LON:
    """
    Local Optima Network (LON) representation.

    A LON is a directed graph where nodes represent local optima and edges
    represent transitions between them discovered during basin-hopping search.

    Attributes:
        graph: The underlying `igraph` Graph object. Default: empty directed `ig.Graph`.
        best_fitness: The best (minimum) fitness value found. Default: `None`.
        final_run_values: `Series` mapping run number to final fitness value. Default: `None`.
        eq_atol: Tolerance for considering fitness values as equal. Default: `1e-12`.
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None
    final_run_values: pd.Series | None = None
    eq_atol: float = DEFAULT_ATOL

    @classmethod
    def from_trace_data(
        cls,
        trace: pd.DataFrame,
        config: LONConfig | None = None,
    ) -> "LON":
        """
        Create a LON from trace data.

        Args:
            trace: DataFrame with columns `[run, fit1, node1, fit2, node2]` where:
                - run: integer run number
                - fit1: fitness value of source node
                - node1: string hash of source node
                - fit2: fitness value of target node
                - node2: string hash of target node
            config: Optional configuration for LON construction. If `None`, uses default
                configuration with minimum fitness aggregation. Default: `None`.

        Returns:
            `LON` instance with constructed graph.

        Raises:
            ValueError: If fitness_aggregation is `"strict"` and duplicates are detected,
                or if `max_fitness_deviation` threshold is exceeded.
        """
        config = config or LONConfig()
        trace = trace.copy()
        trace.columns = pd.Index(["run", "fit1", "node1", "fit2", "node2"])

        # Extract final fitness value from each run as a Series
        final_run_values = trace.groupby("run").tail(1).set_index("run")["fit2"]

        lnodes = pd.concat(
            [
                trace[["node1", "fit1"]].rename(columns={"node1": "Node", "fit1": "Fitness"}),
                trace[["node2", "fit2"]].rename(columns={"node2": "Node", "fit2": "Fitness"}),
            ],
            ignore_index=True,
        )

        # Node deduplication by grouping and aggregation
        node_agg = (
            lnodes.groupby("Node").agg({"Fitness": ["min", "max", "mean", "nunique"]}).reset_index()
        )
        node_agg.columns = pd.Index(
            [
                "Node",
                "Fitness_min",
                "Fitness_max",
                "Fitness_mean",
                "Fitness_nunique",
            ]
        )

        visit_counts = lnodes.groupby("Node").size().reset_index(name="Count")

        duplicates = node_agg[node_agg["Fitness_nunique"] > 1]

        # Check for duplication issues, raise errors or warning according to config
        if not duplicates.empty:
            _validate_duplicate_nodes(node_agg, duplicates, config)

        match config.fitness_aggregation:
            case "first":
                fitness_values = lnodes.groupby("Node", as_index=False).first()[["Node", "Fitness"]]
            case "max":
                fitness_values = node_agg[["Node", "Fitness_max"]].rename(
                    columns={"Fitness_max": "Fitness"}
                )
            case "mean":
                fitness_values = node_agg[["Node", "Fitness_mean"]].rename(
                    columns={"Fitness_mean": "Fitness"}
                )
            case _:  # "min" or default
                fitness_values = node_agg[["Node", "Fitness_min"]].rename(
                    columns={"Fitness_min": "Fitness"}
                )

        # Merge fitness values with visit counts
        nodes = pd.merge(fitness_values, visit_counts, on="Node")

        edges = trace.groupby(["node1", "node2"], as_index=False).size()
        edges.columns = pd.Index(["Start", "End", "Count"])

        graph = ig.Graph(directed=True)

        for _, row in nodes.iterrows():
            graph.add_vertex(name=str(row["Node"]), Fitness=row["Fitness"], Count=row["Count"])

        for _, row in edges.iterrows():
            with contextlib.suppress(ValueError):
                graph.add_edge(str(row["Start"]), str(row["End"]), Count=row["Count"])

        # Remove self-loops
        graph = graph.simplify(multiple=False, loops=True)

        best = nodes["Fitness"].min()

        return cls(
            graph=graph,
            best_fitness=best,
            final_run_values=final_run_values,
            eq_atol=config.eq_atol,
        )

    @property
    def n_vertices(self) -> int:
        """Number of vertices (local optima) in the LON."""
        return int(self.graph.vcount())

    @property
    def n_edges(self) -> int:
        """Number of edges in the LON."""
        return int(self.graph.ecount())

    @property
    def vertex_names(self) -> list[str]:
        """List of vertex names (node hashes)."""
        return list(self.graph.vs["name"])

    @property
    def vertex_fitness(self) -> list[float]:
        """List of vertex fitness values."""
        return list(self.graph.vs["Fitness"])

    @property
    def vertex_count(self) -> list[int]:
        """List of vertex counts (times visited)."""
        return list(self.graph.vs["Count"])

    def _allclose(self, f1: float | None, f2: float | None) -> bool:
        """Check if two fitness values are equal within tolerance."""
        if f1 is None or f2 is None:
            return f1 == f2
        return np.allclose(f1, f2, atol=self.eq_atol, rtol=0.0)

    def _isclose_series(self, f1: pd.Series, f2: float):
        """Check element-wise if a `Series` of fitness values are equal to a scalar within tolerance."""
        return np.isclose(f1, f2, atol=self.eq_atol, rtol=0.0)

    def get_sinks(self) -> list[int]:
        """Get indices of sink nodes (nodes with no outgoing edges)."""
        out_degrees = self.graph.degree(mode="out")
        return [i for i, d in enumerate(out_degrees) if d == 0]

    def compute_network_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute LON network metrics.

        Args:
            known_best: Known global optimum value. If `None`, uses the best
                fitness found in the network. Default: `None`.

        Returns:
            Dictionary containing:
                - n_optima: Number of local optima (vertices)
                - n_funnels: Number of funnels (sinks)
                - n_global_funnels: Number of funnels at global optimum
                - neutral: Proportion of nodes with equal-fitness connections
                - global_strength: Proportion of global optima incoming strength to total incoming strength of all nodes
                - sink_strength: Proportion of global sinks incoming strength to incoming strength of all sink nodes
        """
        best = known_best if known_best is not None else self.best_fitness

        n_optima = self.n_vertices

        sinks_id = self.get_sinks()
        n_funnels = len(sinks_id)

        sinks_fit = [self.vertex_fitness[i] for i in sinks_id]
        n_global_funnels = sum(1 for f in sinks_fit if self._allclose(f, best))

        # Neutral: proportion of nodes with equal-fitness connections
        el = self.graph.get_edgelist()
        fits = self.vertex_fitness
        neutral_edge_indices = []
        for i, (src, tgt) in enumerate(el):
            if self._allclose(fits[src], fits[tgt]):
                neutral_edge_indices.append(i)

        if neutral_edge_indices:
            gnn = self.graph.subgraph_edges(neutral_edge_indices, delete_vertices=True)
            neutral = round(gnn.vcount() / n_optima, 4)
        else:
            neutral = 0.0

        # Strength (global): incoming strength to global optima / total incoming strength
        igs = [i for i, f in enumerate(self.vertex_fitness) if self._allclose(f, best)]
        if self.n_edges > 0 and igs:
            edge_weights = self.graph.es["Count"]
            stren_igs = sum(self.graph.strength(igs, mode="in", loops=False, weights=edge_weights))
            stren_all = sum(self.graph.strength(mode="in", loops=False, weights=edge_weights))
            global_strength = round(stren_igs / stren_all, 4) if stren_all > 0 else 0.0
        else:
            global_strength = 0.0

        # Strength (sinks only): incoming strength to global sinks / incoming strength to all sinks
        global_sinks = [s for s in sinks_id if self._allclose(self.vertex_fitness[s], best)]
        local_sinks = [s for s in sinks_id if not self._allclose(self.vertex_fitness[s], best)]
        if self.n_edges > 0 and global_sinks:
            edge_weights = self.graph.es["Count"]
            sing = sum(
                self.graph.strength(global_sinks, mode="in", loops=False, weights=edge_weights)
            )
            sinl = (
                sum(self.graph.strength(local_sinks, mode="in", loops=False, weights=edge_weights))
                if local_sinks
                else 0
            )
            sink_strength = round(sing / (sing + sinl), 4) if (sing + sinl) > 0 else 0.0
        else:
            sink_strength = 0.0

        return {
            "n_optima": n_optima,
            "n_funnels": n_funnels,
            "n_global_funnels": n_global_funnels,
            "neutral": neutral,
            "global_strength": global_strength,
            "sink_strength": sink_strength,
        }

    def compute_performance_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute performance metrics based on sampling runs.

        Args:
            known_best: Known global optimum value. If `None`, uses the best
                fitness found in the network. Default: `None`.

        Returns:
            Dictionary containing:
                - success: Proportion of runs that reached the global optimum
                - deviation: Mean absolute deviation from the global optimum
        """
        best = known_best if known_best is not None else self.best_fitness
        # Success: proportion of runs that reached the global optimum
        success = (
            self._isclose_series(self.final_run_values, best).sum() / len(self.final_run_values)
            if self.final_run_values is not None and best is not None
            else 0.0
        )

        # Deviation: mean deviation from the global optimum value
        deviation = (
            (self.final_run_values - best).abs().mean()
            if self.final_run_values is not None and best is not None
            else 0.0
        )

        return {"success": success, "deviation": deviation}

    def compute_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute all LON metrics (network topology + performance).

        This is a convenience method that combines both network metrics
        (topology-based) and performance metrics (run-based).

        Args:
            known_best: Known global optimum value. If `None`, uses the best
                fitness found in the network. Default: `None`.

        Returns:
            Dictionary containing all network and performance metrics:
                Network metrics: n_optima, n_funnels, n_global_funnels, neutral, global_strength, sink_strength
                Performance metrics: success, deviation
        """
        network_metrics = self.compute_network_metrics(known_best)
        performance_metrics = self.compute_performance_metrics(known_best)
        return {**network_metrics, **performance_metrics}

    def to_cmlon(self) -> "CMLON":
        """
        Convert LON to Compressed Monotonic LON (CMLON).

        Returns:
            CMLON instance with contracted neutral nodes.
        """
        return CMLON.from_lon(self)


@dataclass
class CMLON:
    """
    Compressed Monotonic Local Optima Network (CMLON).

    CMLON contracts nodes with equal fitness that are connected,
    creating a compressed representation of the fitness landscape.

    Attributes:
        graph: The underlying igraph Graph object.
        best_fitness: The best (minimum) fitness value.
        source_lon: Reference to the original LON (optional).
        eq_atol: Tolerance for considering fitness values as equal. Default: `1e-12`.
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None
    source_lon: LON | None = None
    eq_atol: float = DEFAULT_ATOL

    @classmethod
    def from_lon(cls, lon: LON) -> "CMLON":
        """
        Create CMLON from LON by contracting neutral nodes.

        The compression process:
        1. Mark edges as "improving" (f2 < f1) or "equal" (f2 == f1)
        2. Create subgraph of equal-fitness edges
        3. Find weakly connected components
        4. Contract vertices using component membership
        5. Combine parallel edge weights

        Args:
            lon: Source LON instance.

        Returns:
            CMLON with contracted neutral components.
        """
        if lon.n_edges == 0:
            cmlon_graph = lon.graph.copy()
            return cls(
                graph=cmlon_graph,
                best_fitness=lon.best_fitness,
                source_lon=lon,
                eq_atol=lon.eq_atol,
            )

        # Create a working copy
        mlon = lon.graph.copy()
        mlon.vs["Count"] = [1] * mlon.vcount()

        el = mlon.get_edgelist()
        fits = mlon.vs["Fitness"]

        f1 = [fits[src] for src, _ in el]
        f2 = [fits[tgt] for _, tgt in el]

        # Mark edge types and find equal-fitness edges
        edge_types = []
        equal_edge_indices = []
        for i, (fit1, fit2) in enumerate(zip(f1, f2)):
            if fit2 < fit1:
                edge_types.append("improving")
            elif lon._allclose(fit2, fit1):
                edge_types.append("equal")
                equal_edge_indices.append(i)
            else:
                edge_types.append("worsening")
        mlon.es["type"] = edge_types

        # Create subgraph of equal-fitness edges (keep all vertices)
        if equal_edge_indices:
            gnn = mlon.subgraph_edges(equal_edge_indices, delete_vertices=False)
        else:
            gnn = ig.Graph(n=mlon.vcount(), directed=True)

        # Find weakly connected components
        nn_memb = gnn.components(mode="weak").membership

        # Contract vertices using component membership
        cmlon_graph = _contract_vertices(
            mlon,
            nn_memb,
            vertex_attr_comb={"Fitness": "first", "Count": "sum", "name": "first"},
        )

        return cls(
            graph=cmlon_graph,
            best_fitness=lon.best_fitness,
            source_lon=lon,
            eq_atol=lon.eq_atol,
        )

    def _allclose(self, f1: float | None, f2: float | None) -> bool:
        """Check if two fitness values are equal within tolerance."""
        if f1 is None or f2 is None:
            return f1 == f2
        return np.allclose(f1, f2, atol=self.eq_atol, rtol=0.0)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in CMLON."""
        return int(self.graph.vcount())

    @property
    def n_edges(self) -> int:
        """Number of edges in CMLON."""
        return int(self.graph.ecount())

    @property
    def vertex_fitness(self) -> list[float]:
        """List of vertex fitness values."""
        return list(self.graph.vs["Fitness"])

    @property
    def vertex_count(self) -> list[int]:
        """List of vertex counts (contracted nodes)."""
        return list(self.graph.vs["Count"])

    def get_sinks(self) -> list[int]:
        """Get indices of sink nodes (nodes with no outgoing edges)."""
        out_degrees = self.graph.degree(mode="out")
        return [i for i, d in enumerate(out_degrees) if d == 0]

    def get_global_sinks(self) -> list[int]:
        """Get indices of global sinks (sinks at best fitness)."""
        sinks = self.get_sinks()
        fits = self.vertex_fitness
        return [s for s in sinks if self._allclose(fits[s], self.best_fitness)]

    def get_local_sinks(self) -> list[int]:
        """Get indices of local sinks (sinks not at best fitness)."""
        sinks = self.get_sinks()
        fits = self.vertex_fitness
        if self.best_fitness is None:
            return []
        return [s for s in sinks if fits[s] > self.best_fitness]

    def compute_network_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute CMLON network metrics.

        Args:
            known_best: Known global optimum value. If `None`, uses the best
                fitness found in the network. Default: `None`.

        Returns:
            Dictionary containing:
                - n_optima: Number of optima in CMLON
                - n_funnels: Number of funnels (sinks)
                - n_global_funnels: Number of funnels at global optimum
                - neutral: Proportion of contracted nodes
                - global_strength: Proportion of global sinks incoming strength to total incoming strength of all nodes
                - sink_strength: Proportion of global sinks incoming strength to incoming strength of all sink nodes
                - global_funnel_proportion: Proportion of nodes that can reach
                  a global optimum
        """
        best = known_best if known_best is not None else self.best_fitness

        n_optima = self.n_vertices

        sinks_id = self.get_sinks()
        n_funnels = len(sinks_id)

        sinks_fit = [self.vertex_fitness[i] for i in sinks_id]
        n_global_funnels = sum(1 for f in sinks_fit if self._allclose(f, best))

        # Neutral: proportion of contracted nodes
        if self.source_lon is not None:
            neutral = round(1.0 - self.n_vertices / self.source_lon.n_vertices, 4)
        else:
            neutral = 0.0

        # Strength (global): incoming strength to global sinks / total incoming strength
        igs = [s for s, f in zip(sinks_id, sinks_fit) if self._allclose(f, best)]
        ils = [s for s, f in zip(sinks_id, sinks_fit) if not self._allclose(f, best)]

        if self.n_edges > 0:
            edge_weights = self.graph.es["Count"]
            sing = (
                sum(self.graph.strength(igs, mode="in", loops=False, weights=edge_weights))
                if igs
                else 0
            )
            total = sum(self.graph.strength(mode="in", loops=False, weights=edge_weights))
            global_strength = round(sing / total, 4) if total > 0 else 0.0
        else:
            global_strength = 0.0

        # Strength (sinks only): incoming strength to global sinks / incoming strength to all sinks
        if self.n_edges > 0 and igs:
            edge_weights = self.graph.es["Count"]
            sinl = (
                sum(self.graph.strength(ils, mode="in", loops=False, weights=edge_weights))
                if ils
                else 0
            )
            sink_strength = round(sing / (sing + sinl), 4) if (sing + sinl) > 0 else 0.0
        else:
            sink_strength = 0.0

        gfunnel = self._compute_global_funnel_proportion()

        return {
            "n_optima": n_optima,
            "n_funnels": n_funnels,
            "n_global_funnels": n_global_funnels,
            "neutral": neutral,
            "global_strength": global_strength,
            "sink_strength": sink_strength,
            "global_funnel_proportion": gfunnel,
        }

    def _compute_global_funnel_proportion(self) -> float:
        """Compute proportion of nodes that can reach a global optimum."""
        igs = self.get_global_sinks()
        if not igs:
            return 0.0

        # Get all nodes that can reach any global sink
        reachable = set()
        for sink in igs:
            component = self.graph.subcomponent(sink, mode="in")
            reachable.update(component)

        return len(reachable) / self.n_vertices if self.n_vertices > 0 else 0.0

    def compute_performance_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute performance metrics from the source LON.

        CMLON delegates to its source LON for performance metrics since
        it doesn't have its own sampling run data.

        Args:
            known_best: Known global optimum value. If `None`, uses the best
                fitness found in the network. Default: `None`.

        Returns:
            Dictionary containing performance metrics from source LON, or
            empty dict if no source LON is available.
        """
        return (
            self.source_lon.compute_performance_metrics(known_best)
            if self.source_lon is not None
            else {}
        )

    def compute_metrics(self, known_best: float | None = None) -> dict[str, Any]:
        """
        Compute all CMLON metrics (network topology + performance).

        This is a convenience method that combines both CMLON-specific network
        metrics and performance metrics from the source LON.

        Args:
            known_best: Known global optimum value. If `None`, uses the best
                fitness found in the network. Default: `None`.

        Returns:
            Dictionary containing all network and performance metrics:
                Network metrics: n_optima, n_funnels, n_global_funnels, neutral,
                    global_strength, sink_strength, global_funnel_proportion
                Performance metrics: success, deviation (from source LON)
        """
        network_metrics = self.compute_network_metrics(known_best)
        performance_metrics = self.compute_performance_metrics(known_best)
        return {**network_metrics, **performance_metrics}


def _contract_vertices(
    graph: ig.Graph,
    membership: list[int],
    vertex_attr_comb: dict[str, str],
) -> ig.Graph:
    """
    Contract vertices according to membership, combining attributes.

    Args:
        graph: Input graph.
        membership: Component membership for each vertex.
        vertex_attr_comb: How to combine vertex attributes. Supported methods:
            `"first"`, `"sum"`, `"min"`, `"max"`, `"ignore"`.

    Returns:
        New graph with contracted vertices.
    """
    n_components = max(membership) + 1

    # Group vertices by component
    components: dict[int, list[int]] = {i: [] for i in range(n_components)}
    for v_idx, comp in enumerate(membership):
        components[comp].append(v_idx)

    # Create new graph
    new_graph = ig.Graph(directed=True)
    new_graph.add_vertices(n_components)

    # Combine vertex attributes
    for attr in graph.vs.attributes():
        comb_method = vertex_attr_comb.get(attr, "ignore")
        if comb_method == "ignore":
            continue

        new_values = []
        for comp_idx in range(n_components):
            verts = components[comp_idx]
            values = [graph.vs[v][attr] for v in verts]

            if comb_method == "first":
                new_values.append(values[0] if values else None)
            elif comb_method == "sum":
                new_values.append(sum(values))
            elif comb_method == "min":
                new_values.append(min(values))
            elif comb_method == "max":
                new_values.append(max(values))
            else:
                new_values.append(values[0] if values else None)

        new_graph.vs[attr] = new_values

    # Map old edges to new edges
    new_edges: dict[tuple[int, int], float] = {}
    for edge in graph.es:
        src_comp = membership[edge.source]
        tgt_comp = membership[edge.target]
        if src_comp != tgt_comp:  # Skip self-loops created by contraction
            key = (src_comp, tgt_comp)
            edge_count = edge["Count"] if "Count" in edge.attributes() else 1
            if key in new_edges:
                new_edges[key] += edge_count
            else:
                new_edges[key] = edge_count

    # Add edges
    if new_edges:
        edges = list(new_edges.keys())
        counts = list(new_edges.values())
        new_graph.add_edges(edges)
        new_graph.es["Count"] = counts

    return new_graph


def _validate_duplicate_nodes(
    node_agg: pd.DataFrame,
    duplicates: pd.DataFrame,
    config: LONConfig,
) -> None:
    """
    Validate and warn about nodes that appear with more than one fitness value.

    Args:
        node_agg: Aggregated node DataFrame (columns: Node, Fitness_min, Fitness_max, …).
        duplicates: Subset of node_agg where Fitness_nunique > 1.
        config: LON construction configuration.

    Raises:
        ValueError: If max_fitness_deviation threshold is exceeded, or if
            fitness_aggregation is `"strict"`.
    """
    max_deviation = (node_agg["Fitness_max"] - node_agg["Fitness_min"]).max()

    if config.max_fitness_deviation is not None and max_deviation > config.max_fitness_deviation:
        raise ValueError(
            f"Fitness deviation ({max_deviation:.6f}) exceeds maximum allowed "
            f"threshold ({config.max_fitness_deviation:.6f}). "
            f"Found {len(duplicates)} node(s) with multiple fitness values. "
            f"This may indicate data quality issues."
        )

    if config.fitness_aggregation == "strict":
        raise ValueError(
            f"Detected {len(duplicates)} node(s) with multiple fitness values "
            f"(max deviation: {max_deviation:.6f}) in strict mode. "
            f"Strict mode requires each node to have a unique fitness value. "
            f"Consider using a different fitness_aggregation strategy or "
            f"adjusting coordinate_precision/fitness_precision."
        )

    if config.warn_on_duplicates:
        warnings.warn(
            f"Detected {len(duplicates)} node(s) with multiple fitness values "
            f"(max deviation: {max_deviation:.6f}). "
            f"Using '{config.fitness_aggregation}' fitness for each node. "
            f"This may indicate numerical precision issues or noisy fitness evaluation.",
            category=UserWarning,
            stacklevel=3,
        )
