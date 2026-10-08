import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import igraph as ig
import numpy as np
import pandas as pd

DEFAULT_ATOL = 1e-12
TRACE_COLUMNS = ["run", "fit1", "node1", "fit2", "node2"]


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
        minimize: Whether this is a minimization problem. Default: `True`.
    """

    fitness_aggregation: Literal["min", "max", "mean", "first", "strict"] = "min"
    warn_on_duplicates: bool = True
    max_fitness_deviation: float | None = None
    eq_atol: float = DEFAULT_ATOL
    minimize: bool = True


@dataclass
class LON:
    """
    Local Optima Network (LON) representation.

    A LON is a directed graph where nodes represent local optima and edges
    represent transitions between them discovered during basin-hopping search.

    Attributes:
        graph: The underlying `igraph` Graph object. Default: empty directed `ig.Graph`.
        best_fitness: The best fitness value found. Default: `None`.
        final_run_values: `Series` mapping run number to final fitness value. Default: `None`.
        eq_atol: Tolerance for considering fitness values as equal. Default: `1e-12`.
        minimize: Whether this is a minimization problem. Default: `True`.
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None
    final_run_values: pd.Series | None = None
    eq_atol: float = DEFAULT_ATOL
    minimize: bool = True

    @classmethod
    def from_trace_data(
        cls,
        trace: pd.DataFrame,
        config: LONConfig | None = None,
    ) -> "LON":
        """
        Create a LON from trace data.

        Args:
            trace: DataFrame with one accepted transition per row, in chronological order
                within each run, and columns `[run, fit1, node1, fit2, node2]` (in any order) where:
                - run: integer run number
                - fit1: fitness value of source node
                - node1: string identifier of source node
                - fit2: fitness value of target node
                - node2: string identifier of target node
            config: Optional configuration for LON construction. If `None`, uses default
                configuration with minimum fitness aggregation. Default: `None`.

        Returns:
            `LON` instance with constructed graph.

        Note:
            Edges are kept exactly as recorded, so the LON is not required to be monotonic.
            Worsening edges can appear if the sampler accepts non-improving moves, or when
            the same node is recorded with different fitness values and these are aggregated
            (see `LONConfig.fitness_aggregation`). `CMLON.from_lon()` removes such edges.

            The trace is validated before construction. A `UserWarning` is emitted if a
            trajectory is broken, i.e. within a run `node2` of a row differs from `node1`
            of the next row, which usually means that rows are out of order or include
            rejected moves.

        Raises:
            ValueError: If the trace is malformed: it is empty, its columns differ from
                `[run, fit1, node1, fit2, node2]`, it has missing values, run numbers are
                not integers, fitness values are non-numeric or infinite, or node
                identifiers are not strings. Also raised if fitness_aggregation is `"strict"` and duplicates are
                detected, or if `max_fitness_deviation` threshold is exceeded.
        """
        config = config or LONConfig()
        trace = _validate_trace(trace)

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

        best = nodes["Fitness"].min() if config.minimize else nodes["Fitness"].max()

        return cls(
            graph=graph,
            best_fitness=best,
            final_run_values=final_run_values,
            eq_atol=config.eq_atol,
            minimize=config.minimize,
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
        best_fitness: The best fitness value.
        source_lon: Reference to the original LON (optional).
        eq_atol: Tolerance for considering fitness values as equal. Default: `1e-12`.
        minimize: Whether this is a minimization problem. Default: `True`.
    """

    graph: ig.Graph = field(default_factory=lambda: ig.Graph(directed=True))
    best_fitness: float | None = None
    source_lon: LON | None = None
    eq_atol: float = DEFAULT_ATOL
    minimize: bool = True

    @classmethod
    def from_lon(cls, lon: LON) -> "CMLON":
        """
        Create CMLON from LON by contracting neutral nodes.

        The compression process:
        1. Classify edges using the optimization direction and equality tolerance
        2. Remove worsening edges (with a warning)
        3. Create subgraph of equal-fitness edges
        4. Find weakly connected components
        5. Contract vertices using component membership, keeping the best fitness
           in each component (minimum for minimization, maximum for maximization)
           and the name of the vertex with that fitness (first in graph order on ties)
        6. Combine parallel edge weights
        7. Repeat steps 3-6 until no equal-fitness edges remain. Approximate equality
           is not transitive, so a contracted component can become equal to a
           neighbour it was not equal to before contraction.
        8. Remove edges that became worsening between component representatives
           (with a warning). Such edges cannot be merged, so their target may
           become an isolated sink.

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
                minimize=lon.minimize,
            )

        # Create a working copy
        mlon = lon.graph.copy()
        mlon.vs["Count"] = [1] * mlon.vcount()

        f1, f2 = _edge_fitness(mlon)

        # Mark edge types
        edge_types = []
        for fit1, fit2 in zip(f1, f2):
            if lon._allclose(fit2, fit1):
                edge_types.append("equal")
            elif (fit2 < fit1) if lon.minimize else (fit2 > fit1):
                edge_types.append("improving")
            else:
                edge_types.append("worsening")
        mlon.es["type"] = edge_types

        # Remove worsening edges before compressing neutral components. They come from
        # samplers that accept worse solutions, or from merging duplicate nodes whose
        # fitness values differ slightly.
        worsening_edge_indices = [i for i, t in enumerate(edge_types) if t == "worsening"]
        if worsening_edge_indices:
            max_worsening = max(abs(f2[i] - f1[i]) for i in worsening_edge_indices)
            warnings.warn(
                f"Removed {len(worsening_edge_indices)} worsening edge(s) from the LON "
                f"(max fitness difference: {max_worsening:.6g}) to construct a monotonic CMLON. "
                "If this is unexpected, consider setting `fitness_precision`.",
                UserWarning,
                stacklevel=2,
            )
            mlon.delete_edges(worsening_edge_indices)

        # Contract neutral components until no equal-fitness edges remain. Approximate
        # equality is not transitive, so a contracted component (represented by the
        # best fitness of its vertices) can become equal to a neighbour that was not
        # equal to any of the vertices it was connected to before contraction.
        cmlon_graph = mlon
        while True:
            f_src, f_tgt = _edge_fitness(cmlon_graph)
            equal_edge_indices = np.flatnonzero(
                np.isclose(f_src, f_tgt, atol=lon.eq_atol, rtol=0.0)
            ).tolist()
            if not equal_edge_indices:
                break

            # Find weakly connected components of the equal-fitness subgraph
            gnn = cmlon_graph.subgraph_edges(equal_edge_indices, delete_vertices=False)
            nn_memb = gnn.components(mode="weak").membership

            # Contract vertices using component membership
            cmlon_graph = _contract_vertices(
                cmlon_graph,
                nn_memb,
                vertex_attr_comb={
                    "Fitness": "min" if lon.minimize else "max",
                    "Count": "sum",
                    "name": "min_fitness" if lon.minimize else "max_fitness",
                },
            )

        # Contraction can also turn an improving edge into a worsening one between
        # component representatives: approximate equality is not transitive, so
        # contraction can change an edge's classification. Such edges cannot be
        # merged and are removed.
        f_src, f_tgt = _edge_fitness(cmlon_graph)
        improving = f_tgt < f_src if lon.minimize else f_tgt > f_src
        worsening_after_compression = np.flatnonzero(~improving).tolist()
        if worsening_after_compression:
            warnings.warn(
                f"Removed {len(worsening_after_compression)} worsening edge(s) after "
                "neutral-component compression to construct a monotonic CMLON.",
                UserWarning,
                stacklevel=2,
            )
            cmlon_graph.delete_edges(worsening_after_compression)

        return cls(
            graph=cmlon_graph,
            best_fitness=lon.best_fitness,
            source_lon=lon,
            eq_atol=lon.eq_atol,
            minimize=lon.minimize,
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
        if self.minimize:
            return [s for s in sinks if fits[s] > self.best_fitness]
        return [s for s in sinks if fits[s] < self.best_fitness]

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


def _edge_fitness(graph: ig.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of source and target vertex fitness for each edge, in edge order."""
    fits = graph.vs["Fitness"]
    el = graph.get_edgelist()
    return (
        np.array([fits[src] for src, _ in el], dtype=float),
        np.array([fits[tgt] for _, tgt in el], dtype=float),
    )


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
            `"min_fitness"` / `"max_fitness"` take the attribute from the vertex
            with minimum / maximum Fitness, choosing the first in graph order on ties.

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
            elif comb_method in {"min_fitness", "max_fitness"}:
                select = min if comb_method == "min_fitness" else max
                representative = select(verts, key=lambda v: graph.vs[v]["Fitness"])
                new_values.append(graph.vs[representative][attr])
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


def _validate_trace(trace: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize trace data for `LON.from_trace_data`.

    Returns:
        A copy of the trace with columns `[run, fit1, node1, fit2, node2]`,
        float fitness values and string node identifiers.

    Raises:
        ValueError: If the trace is malformed.
    """
    if trace.columns.duplicated().any() or set(trace.columns) != set(TRACE_COLUMNS):
        raise ValueError(
            f"Trace must have exactly the columns {TRACE_COLUMNS} (in any order), "
            f"got {list(trace.columns)}. Rename the columns of external traces explicitly."
        )
    if trace.empty:
        raise ValueError("Trace is empty.")
    trace = trace[TRACE_COLUMNS].copy()

    missing = trace.isna().any()
    if missing.any():
        raise ValueError(f"Missing values in columns {missing[missing].index.tolist()}.")

    run_dtype = trace["run"].dtype
    if not pd.api.types.is_integer_dtype(run_dtype) or pd.api.types.is_bool_dtype(run_dtype):
        raise ValueError(f"Column 'run' must contain integer run numbers, got dtype {run_dtype}.")

    for col in ["fit1", "fit2"]:
        dtype = trace[col].dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise ValueError(
                f"Column '{col}' must contain numeric fitness values, got dtype {dtype}."
            )
        trace[col] = trace[col].astype(float)
        if np.isinf(trace[col]).any():
            raise ValueError(f"Infinite fitness values in column '{col}'.")

    for col in ["node1", "node2"]:
        if pd.api.types.infer_dtype(trace[col]) != "string":
            raise ValueError(
                f"Column '{col}' must contain string node identifiers. When reading a CSV file, "
                f"use pd.read_csv(..., dtype={{'node1': str, 'node2': str}}) so that identifiers "
                f"such as bitstrings keep their leading zeros."
            )

    next_node1 = trace.groupby("run", sort=False)["node1"].shift(-1)
    broken = next_node1.notna() & (next_node1 != trace["node2"])
    if broken.any():
        warnings.warn(
            f"Detected {int(broken.sum())} broken trajectory step(s): node2 differs from "
            f"node1 of the next row in the same run. Rows within a run must be in "
            f"chronological order and contain only accepted transitions.",
            category=UserWarning,
            stacklevel=3,
        )

    return trace
