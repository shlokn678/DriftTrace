"""Parse an optional user-supplied dependency graph (graph.json) generically.

Accepted formats (both optional; graph.json is never required):

    {"edges": [["feature_a", "feature_b"], ["feature_b", "feature_c"]]}

or the richer declared form:

    {"nodes": {"feature_a": {"kind": "raw_input"},
               "feature_b": {"kind": "derived_feature", "parents": ["feature_a"]}}}

Node kinds are inferred for the edge form: a node with no incoming edge is a
``raw_input``; a node with incoming edges is a ``derived_feature``. The model output
node (:data:`OUTPUT_NODE`) is appended as a ``model_output`` fed by the leaf feature
nodes, so RCA can trace feature drift down to the prediction.

No domain names are assumed; every node name comes from the user's file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drifttrace.bundle.reference import OUTPUT_NODE
from drifttrace.graph.dag import DependencyGraph, GraphValidationError, NodeSpec


class GraphParseError(ValueError):
    """Raised when graph.json is present but malformed. Carries a user-facing message."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


def graph_from_edges(
    edges: list[list[str]],
    *,
    feature_names: list[str] | None = None,
) -> DependencyGraph:
    """Build a :class:`DependencyGraph` from a list of ``[parent, child]`` edges.

    A ``model_output`` node (:data:`OUTPUT_NODE`) is appended, fed by the leaf feature
    nodes (features with no children), so the output participates in RCA. If
    ``feature_names`` is given, edge endpoints are validated against it.
    """
    parents: dict[str, list[str]] = {}
    all_nodes: set[str] = set()
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise GraphParseError(
                "Each graph edge must be a [parent, child] pair.",
                detail=f"invalid edge: {edge!r}",
            )
        parent, child = str(edge[0]), str(edge[1])
        if parent == child:
            raise GraphParseError(f"An edge cannot link a node to itself: '{parent}'.")
        all_nodes.update((parent, child))
        parents.setdefault(child, []).append(parent)
        parents.setdefault(parent, parents.get(parent, []))

    if not all_nodes:
        raise GraphParseError("The dependency graph has no edges.")

    if feature_names is not None:
        known = set(feature_names)
        unknown = sorted(n for n in all_nodes if n not in known)
        if unknown:
            raise GraphParseError(
                "The dependency graph references features not present in the model: "
                + ", ".join(unknown)
                + ".",
            )

    # Leaf feature nodes (no children among features) feed the output node.
    children_of: dict[str, list[str]] = {}
    for child, ps in parents.items():
        for p in ps:
            children_of.setdefault(p, []).append(child)
    leaves = [n for n in all_nodes if not children_of.get(n)]

    specs: list[NodeSpec] = []
    for node in sorted(all_nodes):
        node_parents = tuple(parents.get(node, []))
        kind = "raw_input" if not node_parents else "derived_feature"
        specs.append(NodeSpec(name=node, kind=kind, parents=node_parents))
    # Output node fed by leaf features.
    specs.append(NodeSpec(name=OUTPUT_NODE, kind="model_output", parents=tuple(sorted(leaves))))

    try:
        return DependencyGraph(specs)
    except GraphValidationError as exc:
        raise GraphParseError(
            "The dependency graph is invalid.",
            detail=str(exc),
        ) from exc


def _graph_from_nodes(nodes_spec: dict[str, Any]) -> DependencyGraph:
    """Build from the richer declared ``{"nodes": {...}}`` form."""
    specs: list[NodeSpec] = []
    has_output = False
    for name, spec in nodes_spec.items():
        spec = spec or {}
        kind = spec.get("kind", "derived_feature")
        if kind == "model_output":
            has_output = True
        specs.append(
            NodeSpec(
                name=str(name),
                kind=kind,
                parents=tuple(spec.get("parents", []) or []),
                drift=spec.get("drift", {}) or {},
            )
        )
    if not has_output:
        # Append an output node fed by leaf features.
        names = {s.name for s in specs}
        children: dict[str, list[str]] = {}
        for s in specs:
            for p in s.parents:
                children.setdefault(p, []).append(s.name)
        leaves = [n for n in names if not children.get(n)]
        specs.append(NodeSpec(name=OUTPUT_NODE, kind="model_output", parents=tuple(sorted(leaves))))
    try:
        return DependencyGraph(specs)
    except GraphValidationError as exc:
        raise GraphParseError("The dependency graph is invalid.", detail=str(exc)) from exc


def load_graph_json(
    path: Path,
    *,
    feature_names: list[str] | None = None,
) -> DependencyGraph:
    """Load and validate a graph.json file into a :class:`DependencyGraph`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        raise GraphParseError(
            "The dependency graph file could not be read as JSON.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if isinstance(raw, dict) and raw.get("nodes"):
        return _graph_from_nodes(raw["nodes"])
    if isinstance(raw, dict) and "edges" in raw:
        return graph_from_edges(raw["edges"], feature_names=feature_names)
    raise GraphParseError(
        "The dependency graph must contain an 'edges' list or a 'nodes' map.",
    )
