"""Unit tests for monitoring reports (FR-17)."""

from __future__ import annotations

import json

import pytest

from drifttrace.bundle.graph_json import graph_from_edges
from drifttrace.drift.engine import DriftReport, NodeDriftResult
from drifttrace.drift.verdict import Verdict
from drifttrace.governance.report import MonitoringReport, build_report
from drifttrace.rca.engine import analyze


@pytest.fixture
def graph():
    return graph_from_edges([["feature_a", "feature_b"], ["feature_b", "feature_c"]])


def _drift_report(drifted) -> DriftReport:
    report = DriftReport(window_id="w0", baseline_version="1")
    for n in ("feature_a", "feature_b", "feature_c"):
        v = str(Verdict.DRIFT) if n in drifted else str(Verdict.STABLE)
        report.nodes[n] = NodeDriftResult(
            node=n,
            verdict=v,
            ks={"verdict": v, "p_value": 0.01},
            psi={"verdict": v, "psi": 0.5},
        )
    return report


@pytest.mark.unit
def test_report_json_correct(graph) -> None:
    drift = _drift_report(["feature_a", "feature_b", "feature_c"])
    rca = analyze(drift, graph)
    report = build_report(
        drift, rca, graph, model_version="1", dataset_version="sha256:abc", code_commit="deadbeef"
    )
    d = report.to_dict()
    assert d["model_version"] == "1"
    assert d["dataset_version"] == "sha256:abc"
    assert d["code_commit"] == "deadbeef"
    assert d["classifications"]["feature_a"] == "ROOT_CAUSE"
    assert d["classifications"]["feature_b"] == "SYMPTOM"
    assert d["rca"]["has_root_cause"] is True
    json.dumps(d)  # serializable


@pytest.mark.unit
def test_report_markdown_distinguishes_classes(graph) -> None:
    drift = _drift_report(["feature_a", "feature_b", "feature_c"])
    rca = analyze(drift, graph)
    md = build_report(drift, rca, graph, model_version="1").to_markdown()
    assert "ROOT CAUSE" in md
    assert "feature_a" in md
    assert "symptom path" in md


@pytest.mark.unit
def test_report_no_drift_markdown(graph) -> None:
    drift = _drift_report([])
    rca = analyze(drift, graph)
    md = build_report(drift, rca, graph, model_version="1").to_markdown()
    assert "No root cause detected" in md


@pytest.mark.unit
def test_report_saves_json_md_and_latest(graph, tmp_path) -> None:
    drift = _drift_report(["feature_a"])
    rca = analyze(drift, graph)
    report = build_report(drift, rca, graph, model_version="1")
    paths = report.save(tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["latest"].exists()
    latest = json.loads(paths["latest"].read_text(encoding="utf-8"))
    assert latest["report_id"] == report.report_id


@pytest.mark.unit
def test_report_graph_nodes_follow_dependency_direction_not_feature_order(graph) -> None:
    """Regression: the rendered graph order must follow the declared dependency direction
    (parent -> child), NOT the drift report's feature/insertion order.

    This reproduces the live-RCA screenshot bug: the drift report lists the child
    ``feature_b`` BEFORE its parent ``feature_a``, yet the graph is a -> b -> c. The
    report's ``graph_nodes`` must still be topologically ordered so the UI draws arrows
    root -> symptoms, matching the RCA result (root=feature_a, symptoms b, c).
    """
    # Build a drift report whose dict insertion order puts the child before the parent.
    drift = DriftReport(window_id="w0", baseline_version="1")
    for n in ("feature_b", "feature_a", "feature_c"):  # deliberately NOT topological
        drift.nodes[n] = NodeDriftResult(
            node=n,
            verdict=str(Verdict.DRIFT),
            ks={"verdict": str(Verdict.DRIFT), "p_value": 0.01},
            psi={"verdict": str(Verdict.DRIFT), "psi": 0.5},
        )
    rca = analyze(drift, graph)
    report = build_report(drift, rca, graph, model_version="1")
    d = report.to_dict()

    # graph_nodes is the authoritative topological order (parent before child).
    assert d["graph_nodes"] == ["feature_a", "feature_b", "feature_c"]
    # It excludes the model-output node.
    assert "prediction" not in d["graph_nodes"]
    # And it agrees with the RCA result: the root comes first, symptoms downstream.
    root = d["rca"]["root_cause_candidates"][0]["node"]
    assert root == "feature_a"
    assert d["graph_nodes"].index("feature_a") < d["graph_nodes"].index("feature_b")
    assert d["graph_nodes"].index("feature_b") < d["graph_nodes"].index("feature_c")


@pytest.mark.unit
def test_report_graph_edges_are_declared_edges_not_consecutive_order() -> None:
    """Regression (Task 1): the report must expose the DECLARED edges so the UI draws
    arrows only for real dependencies - never between consecutively-displayed but
    unrelated features. Here feature_a -> feature_b is the only feature edge; an unrelated
    stable feature 'extra' has NO edge and must NOT be connected to anything."""
    from drifttrace.bundle.graph_json import graph_from_edges

    g = graph_from_edges([["feature_a", "feature_b"]])
    drift = DriftReport(window_id="w0", baseline_version="1")
    # Report lists an unrelated stable feature between the chain nodes in dict order.
    for n in ("feature_a", "extra", "feature_b"):
        v = str(Verdict.DRIFT) if n in ("feature_a", "feature_b") else str(Verdict.STABLE)
        drift.nodes[n] = NodeDriftResult(
            node=n, verdict=v, ks={"verdict": v}, psi={"verdict": v, "psi": 0.5}
        )
    rca = analyze(drift, g)
    d = build_report(drift, rca, g, model_version="1").to_dict()

    edges = {tuple(e) for e in d["graph_edges"]}
    # Only the declared feature edge (+ the leaf feature -> output edge) exist.
    assert ("feature_a", "feature_b") in edges
    assert ("feature_b", "prediction") in edges
    # No fabricated edge touches the unrelated stable feature, and no edge is inferred
    # from the dict/display order (e.g. feature_a -> extra or extra -> feature_b).
    assert not any("extra" in e for e in edges)
    assert ("feature_a", "extra") not in edges
    assert ("extra", "feature_b") not in edges


@pytest.mark.unit
def test_report_lifecycle_evidence_present(graph) -> None:
    drift = _drift_report(["feature_a"])
    rca = analyze(drift, graph)
    report: MonitoringReport = build_report(
        drift,
        rca,
        graph,
        model_version="7",
        dataset_version="sha256:xyz",
        code_commit="abc123",
        incident_id="inc-1",
    )
    d = report.to_dict()
    assert d["model_version"] == "7"
    assert d["baseline_version"] == "1"
    assert d["dataset_version"] == "sha256:xyz"
    assert d["code_commit"] == "abc123"
    assert d["incident_id"] == "inc-1"
    assert d["report_id"].startswith("rpt-")
