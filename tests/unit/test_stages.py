"""Unit tests for the orchestration stages (FR-6). Broker-free and Airflow-free."""

from __future__ import annotations

import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.drift.baseline import build_baseline
from drifttrace.graph.loader import load_graph
from drifttrace.orchestration.stages import (
    ApprovalRequired,
    StageResult,
    ValidationFailed,
    drift_check_stage,
    ingest_stage,
    report_stage,
    retrain_stage,
    validate_stage,
)


@pytest.fixture
def baseline_file(tmp_path):
    df = generate(GeneratorParams(n_rows=2000, seed=42))
    graph = load_graph()
    baseline = build_baseline(df, graph, model_version="1")
    path = tmp_path / "baseline.json"
    baseline.save(path)
    return path, df


# ---- ingest ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ingest_stage_writes_dataset(tmp_path) -> None:
    result = ingest_stage(n_rows=300, seed=1, out_dir=tmp_path)
    assert isinstance(result, StageResult)
    assert result.ok
    assert (tmp_path / "dataset.csv").exists()
    assert result.detail["dataset_version"].startswith("sha256:")


# ---- validate -------------------------------------------------------------------------
@pytest.mark.unit
def test_validate_stage_passes_on_good_data(tmp_path) -> None:
    ingest_stage(n_rows=300, seed=1, out_dir=tmp_path)
    result = validate_stage(dataset_path=tmp_path / "dataset.csv")
    assert result.ok
    assert result.detail["passed"] is True


@pytest.mark.unit
def test_validate_stage_raises_on_bad_data(tmp_path) -> None:
    df = generate(GeneratorParams(n_rows=200, seed=1))
    df.loc[0, "credit_score"] = 99999.0  # out of range
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValidationFailed):
        validate_stage(dataset_path=path)


# ---- drift-check ----------------------------------------------------------------------
@pytest.mark.unit
def test_drift_check_no_drift_on_same_distribution(baseline_file) -> None:
    path, df = baseline_file
    result = drift_check_stage(df, baseline_path=path, rel_threshold=0.2)
    assert result.ok
    assert result.detail["drift_detected"] is False
    assert result.detail["drifted_nodes"] == []


@pytest.mark.unit
def test_drift_check_detects_income_shift(baseline_file) -> None:
    path, df = baseline_file
    shifted = df.copy()
    shifted["income"] = shifted["income"] * 12.0  # monthly -> annual
    result = drift_check_stage(shifted, baseline_path=path, rel_threshold=0.2)
    assert result.detail["drift_detected"] is True
    assert "income" in result.detail["drifted_nodes"]


# ---- report ---------------------------------------------------------------------------
@pytest.mark.unit
def test_report_always_writes(baseline_file, tmp_path) -> None:
    path, df = baseline_file
    drift = drift_check_stage(df, baseline_path=path)
    result = report_stage(drift, reports_dir=tmp_path)
    assert result.ok
    assert (tmp_path / "pipeline_report.json").exists()


# ---- retrain approval gate ------------------------------------------------------------
@pytest.mark.unit
def test_retrain_requires_approval() -> None:
    with pytest.raises(ApprovalRequired):
        retrain_stage(approved=False)


@pytest.mark.unit
def test_retrain_gate_blocks_without_approve_flag() -> None:
    """The gate must be explicit: default (no approval) never retrains."""
    with pytest.raises(ApprovalRequired):
        retrain_stage(approved=False, approver="alice")
