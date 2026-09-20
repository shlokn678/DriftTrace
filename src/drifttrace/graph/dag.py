"""The declared dependency graph as a NetworkX DiGraph (FR-8).

The graph is DECLARED in ``config/graph.yaml`` and loaded here; it is never learned
(product/tech steering, FR-8.2). It is the single source of truth for both feature
computation order and RCA upstream traversal (FR-3.4, FR-8.4).

Provides validation (reject cycles, undefined parents, orphans) and traversal
(parents, children, transitive ancestors) per FR-8 AC-1..AC-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

NodeKind = str  # "raw_input" | "derived_feature" | "model_output"
VALID_KINDS = {"raw_input", "derived_feature", "model_output"}


class GraphValidationError(ValueError):
    """Raised when the declared graph is structurally invalid."""


@dataclass(frozen=True)
class NodeSpec:
    """A single node in the declared graph."""

    name: str
    kind: NodeKind
    parents: tuple[str, ...] = ()
    drift: dict = field(default_factory=dict)


class DependencyGraph:
    """Wrapper around a validated NetworkX DiGraph of the feature pipeline."""

    def __init__(self, nodes: list[NodeSpec]) -> None:
        self._nodes = {n.name: n for n in nodes}
        self._graph = self._build_and_validate(nodes)

    # ---- construction / validation -------------------------------------------------
    @staticmethod
    def _build_and_validate(nodes: list[NodeSpec]) -> nx.DiGraph:
        graph: nx.DiGraph = nx.DiGraph()
        names = {n.name for n in nodes}

        for node in nodes:
            if node.kind not in VALID_KINDS:
                raise GraphValidationError(
                    f"node '{node.name}': invalid kind '{node.kind}' "
                    f"(expected one of {sorted(VALID_KINDS)})"
                )
            graph.add_node(node.name, kind=node.kind, drift=node.drift)

        # Edges from declared parents; catch undefined parents.
        undefined: list[str] = []
        for node in nodes:
            for parent in node.parents:
                if parent not in names:
                    undefined.append(f"{parent}->{node.name}")
                else:
                    graph.add_edge(parent, node.name)
        if undefined:
            raise GraphValidationError(f"undefined parent(s) referenced: {sorted(undefined)}")

        # Cycles.
        if not nx.is_directed_acyclic_graph(graph):
            cycle = list(nx.find_cycle(graph))
            raise GraphValidationError(f"graph contains a cycle: {cycle}")

        # Orphans: more than one node but some node has neither parents nor children.
        if graph.number_of_nodes() > 1:
            orphans = [
                n for n in graph.nodes if graph.in_degree(n) == 0 and graph.out_degree(n) == 0
            ]
            if orphans:
                raise GraphValidationError(f"orphan node(s) with no edges: {sorted(orphans)}")

        return graph

    # ---- accessors ------------------------------------------------------------------
    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def nodes(self) -> list[str]:
        return list(self._graph.nodes)

    def spec(self, node: str) -> NodeSpec:
        return self._nodes[node]

    def kind(self, node: str) -> NodeKind:
        return str(self._graph.nodes[node]["kind"])

    def parents(self, node: str) -> list[str]:
        return list(self._graph.predecessors(node))

    def children(self, node: str) -> list[str]:
        return list(self._graph.successors(node))

    def ancestors(self, node: str) -> set[str]:
        """All transitive upstream ancestors of ``node`` (FR-8 AC-3)."""
        return set(nx.ancestors(self._graph, node))

    def descendants(self, node: str) -> set[str]:
        """All transitive downstream descendants of ``node``."""
        return set(nx.descendants(self._graph, node))

    def topological_order(self) -> list[str]:
        """Nodes in dependency order (parents before children)."""
        return list(nx.topological_sort(self._graph))

    def feature_nodes(self) -> list[str]:
        """Raw inputs and derived features (drift-checkable), in topological order."""
        return [n for n in self.topological_order() if self.kind(n) != "model_output"]
