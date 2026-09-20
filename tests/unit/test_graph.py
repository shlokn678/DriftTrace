"""Unit tests for the declared dependency graph (FR-8) and feature/graph consistency (FR-3.2)."""

from __future__ import annotations

import pytest

from drifttrace.features.transform import DERIVED_FEATURES, RAW_INPUTS
from drifttrace.graph.dag import DependencyGraph, GraphValidationError, NodeSpec
from drifttrace.graph.loader import load_graph


@pytest.mark.unit
def test_loads_declared_chain() -> None:
    g = load_graph()
    assert set(g.nodes) == {"income", "credit_score", "risk_score", "prediction"}
    assert g.kind("income") == "raw_input"
    assert g.kind("prediction") == "model_output"


@pytest.mark.unit
def test_traversal_parents_children_ancestors() -> None:
    g = load_graph()
    assert g.parents("credit_score") == ["income"]
    assert g.children("income") == ["credit_score"]
    assert g.ancestors("risk_score") == {"income", "credit_score"}
    assert g.ancestors("prediction") == {"income", "credit_score", "risk_score"}


@pytest.mark.unit
def test_topological_order_parents_before_children() -> None:
    g = load_graph()
    order = g.topological_order()
    assert order.index("income") < order.index("credit_score")
    assert order.index("credit_score") < order.index("risk_score")
    assert order.index("risk_score") < order.index("prediction")


@pytest.mark.unit
def test_feature_nodes_exclude_model_output() -> None:
    g = load_graph()
    assert "prediction" not in g.feature_nodes()
    assert g.feature_nodes() == ["income", "credit_score", "risk_score"]


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
        NodeSpec("income", "raw_input"),
        NodeSpec("credit_score", "derived_feature", parents=("income",)),
        NodeSpec("lonely", "raw_input"),
    ]
    with pytest.raises(GraphValidationError, match="orphan"):
        DependencyGraph(nodes)


@pytest.mark.unit
def test_invalid_kind_rejected() -> None:
    with pytest.raises(GraphValidationError, match="invalid kind"):
        DependencyGraph([NodeSpec("x", "not_a_kind")])


@pytest.mark.unit
def test_feature_graph_consistency() -> None:
    """FR-3.2 / FR-3 AC-2: features computed in code must match the declared graph.

    The transform's raw inputs + derived features must be exactly the graph's
    feature nodes (everything except the model output).
    """
    g = load_graph()
    graph_feature_nodes = set(g.feature_nodes())
    code_feature_nodes = set(RAW_INPUTS) | set(DERIVED_FEATURES)
    assert code_feature_nodes == graph_feature_nodes, (
        f"code features {sorted(code_feature_nodes)} != "
        f"graph features {sorted(graph_feature_nodes)}"
    )
