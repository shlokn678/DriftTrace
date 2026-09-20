"""Load the declared dependency graph from YAML into a DependencyGraph (FR-8.4)."""

from __future__ import annotations

from pathlib import Path

import yaml

from drifttrace.config import get_paths
from drifttrace.graph.dag import DependencyGraph, GraphValidationError, NodeSpec


def load_graph(path: Path | None = None) -> DependencyGraph:
    """Load and validate the dependency graph from ``config/graph.yaml``.

    Raises :class:`GraphValidationError` if the graph is structurally invalid
    (cycle, undefined parent, orphan) per FR-8 AC-2.
    """
    graph_path = path or get_paths().graph_yaml
    raw = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
    if not raw or "nodes" not in raw:
        raise GraphValidationError(f"graph file {graph_path} has no 'nodes' section")

    nodes: list[NodeSpec] = []
    for name, spec in raw["nodes"].items():
        spec = spec or {}
        nodes.append(
            NodeSpec(
                name=name,
                kind=spec.get("kind", ""),
                parents=tuple(spec.get("parents", []) or []),
                drift=spec.get("drift", {}) or {},
            )
        )
    return DependencyGraph(nodes)
