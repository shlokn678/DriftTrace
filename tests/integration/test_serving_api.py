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
        reports_dir_path=tmp_path / "reports",
        cors_allow_origins=["http://localhost:5173"],
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


# --- Phase 4 endpoints -----------------------------------------------------------------
@pytest.mark.integration
def test_metrics_endpoint_prometheus_format(client) -> None:
    c, _ = client
    # Make a prediction so a counter increments.
    c.post("/predict", json={"income": 3000.0, "request_id": "m1"})
    resp = c.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "drifttrace_prediction_requests_total" in body
    assert "# TYPE drifttrace_prediction_requests_total counter" in body


@pytest.mark.integration
def test_rca_latest_not_available_initially(client) -> None:
    c, _ = client
    resp = c.get("/rca/latest")
    assert resp.status_code == 200
    # No monitoring cycle has written a report in this isolated temp reports dir.
    assert resp.json()["available"] is False


@pytest.mark.integration
def test_rca_latest_returns_persisted_report(trained_env) -> None:
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    settings, _ = trained_env
    # Write a latest_rca.json into the configured reports dir.
    settings.reports_dir_path.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir_path / "latest_rca.json").write_text(
        '{"report_id": "rpt-test", "rca": {"has_root_cause": true}}', encoding="utf-8"
    )
    with TestClient(create_app(settings)) as c:
        body = c.get("/rca/latest").json()
        assert body["available"] is True
        assert body["report"]["report_id"] == "rpt-test"


@pytest.mark.integration
def test_explain_endpoint_shap(client) -> None:
    c, _ = client
    resp = c.post("/explain", json={"income": 4000.0, "method": "shap"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "shap"
    assert body["model_version"] is not None
    assert len(body["attributions"]) == 3


@pytest.mark.integration
def test_explain_off_hot_path_predict_has_no_attributions(client) -> None:
    c, _ = client
    predict = c.post("/predict", json={"income": 4000.0}).json()
    assert "attributions" not in predict  # /predict never computes explanations


# --- Demo scenario endpoints (real Phase 4 pipeline over HTTP) -------------------------
@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    """Full temp root: dataset + trained model + baseline + config, via DRIFTTRACE_ROOT."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    (tmp_path / "config").mkdir()
    from drifttrace.config import _repo_root

    for name in ["schema.yaml", "graph.yaml", "drift.yaml", "governance.yaml"]:
        src = _repo_root() / "config" / name
        (tmp_path / "config" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    from drifttrace.data.build import build_dataset
    from drifttrace.training.pipeline import run_training_pipeline
    from drifttrace.training.train import TrainConfig

    build_dataset(n_rows=1500, seed=42, out_dir=tmp_path / "data")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_training_pipeline(
        dataset_path=tmp_path / "data" / "dataset.csv",
        config=TrainConfig(seed=42, min_roc_auc=0.6),
        tracking_uri=tracking_uri,
        artifacts_dir=tmp_path / "artifacts",
    )
    from drifttrace.serving.config import get_serving_settings

    return get_serving_settings()


@pytest.mark.integration
def test_demo_scenarios_lists_four(demo_env) -> None:
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    with TestClient(create_app(demo_env)) as c:
        body = c.get("/demo/scenarios").json()
        assert set(body["scenarios"]) == {"control", "income_annual", "mid_chain", "two_roots"}


@pytest.mark.integration
def test_demo_control_no_root_cause(demo_env) -> None:
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    with TestClient(create_app(demo_env)) as c:
        body = c.post("/demo/run-scenario", json={"scenario": "control", "n": 300}).json()
        assert body["scenario"] == "control"
        assert body["outcome"]["has_root_cause"] is False
        assert body["outcome"]["root_cause_candidates"] == []


@pytest.mark.integration
def test_demo_income_annual_root_cause_income(demo_env) -> None:
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    with TestClient(create_app(demo_env)) as c:
        body = c.post("/demo/run-scenario", json={"scenario": "income_annual", "n": 300}).json()
        out = body["outcome"]
        assert "income" in out["drifted_nodes"]
        assert out["root_cause_candidates"] == ["income"]
        assert set(out["symptoms"]).issubset({"credit_score", "risk_score"})
        # After running, /rca/latest reflects the persisted report.
        latest = c.get("/rca/latest").json()
        assert latest["available"] is True
        assert latest["report"]["rca"]["has_root_cause"] is True


@pytest.mark.integration
def test_demo_unknown_scenario_422(demo_env) -> None:
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    with TestClient(create_app(demo_env)) as c:
        resp = c.post("/demo/run-scenario", json={"scenario": "nope", "n": 300})
        assert resp.status_code == 422
