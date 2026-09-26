"""Integration tests for the Phase 2 CLI DAG-stage commands (FR-6.4)."""

from __future__ import annotations

import pytest

from drifttrace.cli.main import main
from drifttrace.config import _repo_root


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """Point DRIFTTRACE_ROOT at a temp dir with a copy of the real config."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    (tmp_path / "config").mkdir()
    for name in ["schema.yaml", "graph.yaml", "drift.yaml", "governance.yaml"]:
        src = _repo_root() / "config" / name
        (tmp_path / "config" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


@pytest.mark.integration
def test_cli_ingest_then_validate(temp_root) -> None:
    assert main(["ingest", "--n-rows", "300", "--seed", "1"]) == 0
    assert (temp_root / "data" / "dataset.csv").exists()
    assert main(["validate"]) == 0


@pytest.mark.integration
def test_cli_validate_fails_on_bad_data(temp_root) -> None:
    from drifttrace.data.generator import GeneratorParams, generate

    (temp_root / "data").mkdir()
    df = generate(GeneratorParams(n_rows=200, seed=1))
    df.loc[0, "risk_score"] = 5.0  # out of [0,1]
    df.to_csv(temp_root / "data" / "dataset.csv", index=False)
    assert main(["validate"]) == 1  # non-zero -> blocks downstream


@pytest.mark.integration
def test_cli_retrain_without_approve_returns_gate_code(temp_root) -> None:
    # Retrain must refuse without --approve (exit code 2), never silently retrain.
    assert main(["retrain"]) == 2


@pytest.mark.integration
def test_cli_drift_check_and_report(temp_root) -> None:
    # Build dataset + a baseline via the training pipeline against a temp MLflow store.
    from drifttrace.training.pipeline import run_training_pipeline
    from drifttrace.training.train import TrainConfig

    main(["ingest", "--n-rows", "1500", "--seed", "42"])
    tracking_uri = f"sqlite:///{(temp_root / 'mlflow.db').as_posix()}"
    run_training_pipeline(
        dataset_path=temp_root / "data" / "dataset.csv",
        config=TrainConfig(seed=42, min_roc_auc=0.6),
        tracking_uri=tracking_uri,
        artifacts_dir=temp_root / "artifacts",
    )
    assert main(["drift-check"]) == 0
    assert main(["report"]) == 0
    assert (temp_root / "reports" / "pipeline_report.json").exists()


@pytest.mark.integration
def test_cli_run_dag_full_sequence(temp_root) -> None:
    from drifttrace.training.pipeline import run_training_pipeline
    from drifttrace.training.train import TrainConfig

    main(["ingest", "--n-rows", "1500", "--seed", "42"])
    tracking_uri = f"sqlite:///{(temp_root / 'mlflow.db').as_posix()}"
    run_training_pipeline(
        dataset_path=temp_root / "data" / "dataset.csv",
        config=TrainConfig(seed=42, min_roc_auc=0.6),
        tracking_uri=tracking_uri,
        artifacts_dir=temp_root / "artifacts",
    )
    # Without --approve, run-dag completes but retrain is skipped (gate).
    assert main(["run-dag"]) == 0
    assert (temp_root / "reports" / "pipeline_report.json").exists()
