"""Unit tests for the graph-based RCA engine (generic, any feature names)."""

from __future__ import annotations

import pytest

from drifttrace.bundle.graph_json import graph_from_edges
from drifttrace.drift.engine import DriftReport, NodeDriftResult
from drifttrace.drift.verdict import Verdict
from drifttrace.graph.dag import DependencyGraph, NodeSpec
from drifttrace.rca.engine import analyze, classify_nodes

CHAIN = ("feature_a", "feature_b", "feature_c")


@pytest.fixture
def graph():
    # feature_a -> feature_b -> feature_c (+ prediction output).
    return graph_from_edges([["feature_a", "feature_b"], ["feature_b", "feature_c"]])


def _report(drifted: list[str], all_nodes=CHAIN) -> DriftReport:
    report = DriftReport(window_id="w0", baseline_version="1")
    for n in all_nodes:
        verdict = str(Verdict.DRIFT) if n in drifted else str(Verdict.STABLE)
        report.nodes[n] = NodeDriftResult(node=n, verdict=verdict, ks={}, psi={"psi": 0.5})
    return report


@pytest.mark.unit
def test_no_drift_no_root_cause(graph) -> None:
    rca = analyze(_report([]), graph)
    assert rca.has_root_cause is False
    assert rca.root_cause_candidates == []


@pytest.mark.unit
def test_upstream_root_cause_with_downstream_symptoms(graph) -> None:
    rca = analyze(_report(["feature_a", "feature_b", "feature_c"]), graph)
    assert rca.has_root_cause is True
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["feature_a"]
    assert set(rca.symptoms) == {"feature_b", "feature_c"}
    cand = rca.root_cause_candidates[0]
    assert cand.symptom_path == ["feature_b", "feature_c"]
    assert set(cand.evidence.keys()) == {"feature_a", "feature_b", "feature_c"}


@pytest.mark.unit
def test_mid_chain_root(graph) -> None:
    rca = analyze(_report(["feature_b", "feature_c"]), graph)
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["feature_b"]
    assert rca.symptoms == ["feature_c"]


@pytest.mark.unit
def test_single_leaf_drift(graph) -> None:
    rca = analyze(_report(["feature_c"]), graph)
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["feature_c"]
    assert rca.symptoms == []


@pytest.mark.unit
def test_upstream_and_leaf_drift_but_middle_stable_is_still_single_root(graph) -> None:
    # In a linear chain feature_a is a transitive ancestor of feature_c, so feature_c is
    # a SYMPTOM of feature_a even when feature_b is stable (single root).
    rca = analyze(_report(["feature_a", "feature_c"]), graph)
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["feature_a"]
    assert rca.symptoms == ["feature_c"]


@pytest.mark.unit
def test_two_independent_roots_on_branched_graph() -> None:
    # Co-equal roots require independent source branches:  a -> c ; b -> c.
    branched = DependencyGraph(
        [
            NodeSpec("a", "raw_input"),
            NodeSpec("b", "raw_input"),
            NodeSpec("c", "derived_feature", parents=("a", "b")),
        ]
    )
    report = DriftReport(window_id="w0", baseline_version="1")
    for n in ("a", "b", "c"):
        report.nodes[n] = NodeDriftResult(
            node=n, verdict=str(Verdict.DRIFT), ks={}, psi={"psi": 0.5}
        )
    rca = analyze(report, branched)
    roots = {c.node for c in rca.root_cause_candidates}
    assert roots == {"a", "b"}
    assert rca.symptoms == ["c"]
    assert len(rca.root_cause_candidates) == 2


@pytest.mark.unit
def test_classify_nodes(graph) -> None:
    cls = classify_nodes(_report(["feature_a", "feature_b", "feature_c"]), graph)
    assert cls["feature_a"] == "ROOT_CAUSE"
    assert cls["feature_b"] == "SYMPTOM"
    assert cls["feature_c"] == "SYMPTOM"


@pytest.mark.unit
def test_rca_serializable(graph) -> None:
    import json

    rca = analyze(_report(["feature_a", "feature_b", "feature_c"]), graph)
    json.dumps(rca.to_dict())
