"""Integration tests for the dataset build and training pipeline (FR-1, FR-6.4, FR-17)."""

from __future__ import annotations

import json

import pytest

from drifttrace.data.build import build_dataset
from drifttrace.training.pipeline import run_training_pipeline
from drifttrace.training.train import TrainConfig


@pytest.mark.integration
def test_build_dataset_writes_artifacts(tmp_path) -> None:
    manifest = build_dataset(n_rows=800, seed=42, out_dir=tmp_path)
    assert (tmp_path / "dataset.csv").exists()
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "dataset_version.txt").exists()
    assert manifest["validation"]["passed"] is True
    assert manifest["dataset_version"].startswith("sha256:")


@pytest.mark.integration
def test_build_dataset_is_deterministic(tmp_path) -> None:
    m1 = build_dataset(n_rows=800, seed=42, out_dir=tmp_path / "a")
    m2 = build_dataset(n_rows=800, seed=42, out_dir=tmp_path / "b")
    assert m1["dataset_version"] == m2["dataset_version"]


@pytest.mark.integration
def test_training_pipeline_end_to_end(tmp_path) -> None:
    # Materialize a dataset, then run the full pipeline against a temp MLflow store.
    build_dataset(n_rows=2500, seed=42, out_dir=tmp_path / "data")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    result = run_training_pipeline(
        dataset_path=tmp_path / "data" / "dataset.csv",
        config=TrainConfig(seed=42, min_roc_auc=0.6),
        tracking_uri=tracking_uri,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result.passed_gate
    assert result.model_version is not None
    assert result.roc_auc > 0.5

    # Lifecycle evidence is complete (FR-17): all five anchors present/derivable.
    ev = result.evidence
    assert ev.dataset_version.startswith("sha256:")
    assert ev.code_commit  # 'unknown' if git missing, but present
    assert ev.pipeline_execution_id
    assert ev.model_version is not None

    # Baseline + run manifest were persisted.
    baseline_path = tmp_path / "artifacts" / "baseline.json"
    manifest_path = tmp_path / "artifacts" / "run_manifest.json"
    assert baseline_path.exists()
    assert manifest_path.exists()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert set(baseline["nodes"].keys()) == {"income", "credit_score", "risk_score"}


@pytest.mark.integration
def test_cli_build_dataset(tmp_path, monkeypatch) -> None:
    # Point the repo data dir at a temp location via DRIFTTRACE_ROOT.
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    (tmp_path / "config").mkdir()
    # Copy the real schema so validation can run.
    from drifttrace.config import _repo_root

    schema_src = _repo_root() / "config" / "schema.yaml"
    (tmp_path / "config" / "schema.yaml").write_text(
        schema_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    from drifttrace.cli.main import main

    rc = main(["build-dataset", "--n-rows", "300", "--seed", "1"])
    assert rc == 0
    assert (tmp_path / "data" / "dataset.csv").exists()
