"""Integration tests for the FastAPI serving app + event emission (FR-7, FR-7.4)."""

from __future__ import annotations

import json

import pytest

from drifttrace.data.build import build_dataset
from drifttrace.serving.config import ServingSettings
from drifttrace.training.pipeline import run_training_pipeline
from drifttrace.training.train import TrainConfig


@pytest.fixture
def trained_env(tmp_path):
    """Materialize a dataset + trained model in a temp MLflow store; return settings."""
    build_dataset(n_rows=1500, seed=42, out_dir=tmp_path / "data")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_training_pipeline(
        dataset_path=tmp_path / "data" / "dataset.csv",
        config=TrainConfig(seed=42, min_roc_auc=0.6),
        tracking_uri=tracking_uri,
        artifacts_dir=tmp_path / "artifacts",
    )
    event_log = tmp_path / "events.jsonl"
    settings = ServingSettings(
        mlflow_tracking_uri=tracking_uri,
        model_version=None,
        use_redpanda=False,
        redpanda_brokers="redpanda:9092",
        topic="drifttrace.predictions",
        event_log_path=event_log,
        transform_params_path=tmp_path / "artifacts" / "transform_params.json",
    )
    return settings, event_log


@pytest.fixture
def client(trained_env):
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    settings, event_log = trained_env
    app = create_app(settings)
    with TestClient(app) as c:
        yield c, event_log


@pytest.mark.integration
def test_health(client) -> None:
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.integration
def test_ready_after_model_load(client) -> None:
    c, _ = client
    resp = c.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["model_version"] is not None


@pytest.mark.integration
def test_model_info(client) -> None:
    c, _ = client
    body = c.get("/model-info").json()
    assert body["model_name"] == "drifttrace-loan-default"
    assert body["features"] == ["income", "credit_score", "risk_score"]
    assert body["loaded"] is True


@pytest.mark.integration
def test_predict_returns_prediction_and_version(client) -> None:
    c, _ = client
    resp = c.post("/predict", json={"income": 4000.0, "request_id": "r1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] is not None
    assert set(body["features"]) == {"income", "credit_score", "risk_score"}


@pytest.mark.integration
def test_predict_emits_event(client) -> None:
    c, event_log = client
    resp = c.post("/predict", json={"income": 5000.0, "request_id": "r2"})
    assert resp.json()["event_emitted"] is True
    assert event_log.exists()
    lines = [ln for ln in event_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1
    ev = json.loads(lines[-1])
    assert ev["request_id"] == "r2"
    assert ev["model_version"] is not None
    assert ev["source"] == "api"
    assert set(ev["features"]) == {"income", "credit_score", "risk_score"}


@pytest.mark.integration
def test_predict_malformed_returns_422(client) -> None:
    c, _ = client
    # Negative income violates ge=0 -> 422, and the service must not crash.
    resp = c.post("/predict", json={"income": -5.0})
    assert resp.status_code == 422
    assert c.get("/health").status_code == 200
