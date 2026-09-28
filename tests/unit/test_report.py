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
