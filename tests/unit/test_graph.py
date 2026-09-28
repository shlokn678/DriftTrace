"""Unit tests for the declared dependency graph (generic, any feature names)."""

from __future__ import annotations

import pytest

from drifttrace.bundle.graph_json import graph_from_edges
from drifttrace.graph.dag import DependencyGraph, GraphValidationError, NodeSpec


def _chain() -> DependencyGraph:
    # feature_a -> feature_b -> feature_c (+ an appended prediction output node).
    return graph_from_edges([["feature_a", "feature_b"], ["feature_b", "feature_c"]])


@pytest.mark.unit
def test_builds_chain_from_edges() -> None:
    g = _chain()
    assert set(g.nodes) == {"feature_a", "feature_b", "feature_c", "prediction"}
    assert g.kind("feature_a") == "raw_input"
    assert g.kind("prediction") == "model_output"


@pytest.mark.unit
def test_traversal_parents_children_ancestors() -> None:
    g = _chain()
    assert g.parents("feature_b") == ["feature_a"]
    assert g.children("feature_a") == ["feature_b"]
    assert g.ancestors("feature_c") == {"feature_a", "feature_b"}
    assert {"feature_a", "feature_b", "feature_c"} <= g.ancestors("prediction")


@pytest.mark.unit
def test_topological_order_parents_before_children() -> None:
    g = _chain()
    order = g.topological_order()
    assert order.index("feature_a") < order.index("feature_b")
    assert order.index("feature_b") < order.index("feature_c")
    assert order.index("feature_c") < order.index("prediction")


@pytest.mark.unit
def test_feature_nodes_exclude_model_output() -> None:
    g = _chain()
    assert "prediction" not in g.feature_nodes()
    assert set(g.feature_nodes()) == {"feature_a", "feature_b", "feature_c"}


@pytest.mark.unit
def test_cycle_rejected() -> None:
    nodes = [
        NodeSpec("a", "raw_input", parents=("c",)),
        NodeSpec("b", "derived_feature", parents=("a",)),
        NodeSpec("c", "derived_feature", parents=("b",)),
    ]
    with pytest.raises(GraphValidationError, match="cycle"):
        DependencyGraph(nodes)


@pytest.mark.unit
def test_undefined_parent_rejected() -> None:
    nodes = [NodeSpec("a", "derived_feature", parents=("ghost",))]
    with pytest.raises(GraphValidationError, match="undefined parent"):
        DependencyGraph(nodes)


@pytest.mark.unit
def test_orphan_rejected() -> None:
    nodes = [
        NodeSpec("a", "raw_input"),
        NodeSpec("b", "derived_feature", parents=("a",)),
        NodeSpec("lonely", "raw_input"),
    ]
    with pytest.raises(GraphValidationError, match="orphan"):
        DependencyGraph(nodes)


@pytest.mark.unit
def test_invalid_kind_rejected() -> None:
    with pytest.raises(GraphValidationError, match="invalid kind"):
        DependencyGraph([NodeSpec("x", "not_a_kind")])
