"""Unit tests for the graph-based RCA engine (FR-10)."""

from __future__ import annotations

import pytest

from drifttrace.drift.engine import DriftReport, NodeDriftResult
from drifttrace.drift.verdict import Verdict
from drifttrace.graph.loader import load_graph
from drifttrace.rca.engine import analyze, classify_nodes


@pytest.fixture
def graph():
    return load_graph()


def _report(drifted: list[str], all_nodes=("income", "credit_score", "risk_score")) -> DriftReport:
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
def test_income_root_cause_with_downstream_symptoms(graph) -> None:
    # The main pitch scenario: all three chained nodes drift.
    rca = analyze(_report(["income", "credit_score", "risk_score"]), graph)
    assert rca.has_root_cause is True
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["income"]
    assert set(rca.symptoms) == {"credit_score", "risk_score"}
    # Symptom path is downstream toward the output.
    cand = rca.root_cause_candidates[0]
    assert cand.symptom_path == ["credit_score", "risk_score"]
    # Evidence attached for the root + path nodes.
    assert set(cand.evidence.keys()) == {"income", "credit_score", "risk_score"}


@pytest.mark.unit
def test_mid_chain_root(graph) -> None:
    # income stable; credit_score + risk_score drift -> credit_score is the earliest root.
    rca = analyze(_report(["credit_score", "risk_score"]), graph)
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["credit_score"]
    assert rca.symptoms == ["risk_score"]


@pytest.mark.unit
def test_single_leaf_drift(graph) -> None:
    rca = analyze(_report(["risk_score"]), graph)
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["risk_score"]
    assert rca.symptoms == []


@pytest.mark.unit
def test_income_and_risk_drift_but_credit_stable_is_still_single_root(graph) -> None:
    # In the LINEAR production chain, income is a transitive ancestor of risk_score, so
    # even if credit_score is stable, risk_score is a SYMPTOM of income (single root).
    rca = analyze(_report(["income", "risk_score"]), graph)
    roots = [c.node for c in rca.root_cause_candidates]
    assert roots == ["income"]
    assert rca.symptoms == ["risk_score"]


@pytest.mark.unit
def test_two_independent_roots_on_branched_graph() -> None:
    # Co-equal roots require independent source branches. Build a branched graph:
    #   a -> c ;  b -> c   (a and b are independent sources feeding c)
    from drifttrace.graph.dag import DependencyGraph, NodeSpec

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
    assert roots == {"a", "b"}  # both independent roots reported
    assert rca.symptoms == ["c"]
    assert len(rca.root_cause_candidates) == 2


@pytest.mark.unit
def test_classify_nodes(graph) -> None:
    cls = classify_nodes(_report(["income", "credit_score", "risk_score"]), graph)
    assert cls["income"] == "ROOT_CAUSE"
    assert cls["credit_score"] == "SYMPTOM"
    assert cls["risk_score"] == "SYMPTOM"


@pytest.mark.unit
def test_rca_serializable(graph) -> None:
    import json

    rca = analyze(_report(["income", "credit_score", "risk_score"]), graph)
    json.dumps(rca.to_dict())
