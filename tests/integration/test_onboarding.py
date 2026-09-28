"""Integration tests for model onboarding endpoints (Phase 5)."""

from __future__ import annotations

import io

import cloudpickle
import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.training.train import TrainConfig, train


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Temp project root with a trained model + baseline so onboarding sees a reference."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    (tmp_path / "config").mkdir()
    from drifttrace.config import _repo_root

    for name in ["schema.yaml", "graph.yaml", "drift.yaml", "governance.yaml"]:
        src = _repo_root() / "config" / name
        (tmp_path / "config" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    from drifttrace.data.build import build_dataset
    from drifttrace.training.pipeline import run_training_pipeline

    build_dataset(n_rows=1200, seed=42, out_dir=tmp_path / "data")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_training_pipeline(
        dataset_path=tmp_path / "data" / "dataset.csv",
        config=TrainConfig(seed=42, min_roc_auc=0.6),
        tracking_uri=tracking_uri,
        artifacts_dir=tmp_path / "artifacts",
    )
    from drifttrace.serving.config import get_serving_settings

    return get_serving_settings()


@pytest.fixture
def client(app_env):
    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app

    # Reset the process-wide onboarding registry between tests.
    from drifttrace.serving.onboarding import REGISTRY

    REGISTRY._models.clear()
    REGISTRY.deactivate()
    return TestClient(create_app(app_env))


def _model_bytes() -> bytes:
    frame = generate(GeneratorParams(n_rows=800, seed=1))
    estimator, _ = train(frame, TrainConfig(seed=1, min_roc_auc=0.6))
    buf = io.BytesIO()
    cloudpickle.dump(estimator, buf)
    return buf.getvalue()


@pytest.mark.integration
def test_upload_supported_model_ready(client) -> None:
    files = {"file": ("model.pkl", _model_bytes(), "application/octet-stream")}
    resp = client.post("/models/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["supported"] is True
    assert body["framework"] == "scikit-learn"
    assert body["n_features"] == 3
    # Reference + dependency graph exist in this env -> ready and nothing missing.
    assert body["reference_available"] is True
    assert body["dependencies_available"] is True
    assert body["ready_to_monitor"] is True
    assert body["missing"] == []
    assert body["model_id"]


@pytest.mark.integration
def test_upload_unsupported_extension(client) -> None:
    files = {"file": ("model.onnx", b"not a model", "application/octet-stream")}
    resp = client.post("/models/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["supported"] is False
    assert body["model_id"] is None
    assert body["message"]


@pytest.mark.integration
def test_upload_non_model_pickle(client) -> None:
    buf = io.BytesIO()
    cloudpickle.dump([1, 2, 3], buf)
    files = {"file": ("thing.pkl", buf.getvalue(), "application/octet-stream")}
    resp = client.post("/models/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["supported"] is False


@pytest.mark.integration
def test_model_status_after_upload(client) -> None:
    files = {"file": ("model.pkl", _model_bytes(), "application/octet-stream")}
    model_id = client.post("/models/upload", files=files).json()["model_id"]
    status = client.get(f"/models/{model_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["model_id"] == model_id
    assert body["supported"] is True
    assert body["ready_to_monitor"] is True


@pytest.mark.integration
def test_model_status_not_found(client) -> None:
    assert client.get("/models/does-not-exist").status_code == 404


@pytest.mark.integration
def test_upload_requires_file(client) -> None:
    assert client.post("/models/upload").status_code == 422


# ---- Phase 5: active-model workflow ---------------------------------------------------
@pytest.mark.integration
def test_active_model_defaults_to_loan_model(client) -> None:
    resp = client.get("/models/active")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_custom"] is False
    assert body["name"] == "drifttrace-loan-default"
    assert body["model_id"] is None


@pytest.mark.integration
def test_activate_uploaded_model_becomes_active(client) -> None:
    files = {"file": ("model.pkl", _model_bytes(), "application/octet-stream")}
    model_id = client.post("/models/upload", files=files).json()["model_id"]

    resp = client.post(f"/models/{model_id}/activate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_custom"] is True
    assert body["model_id"] == model_id

    # /models/active and /model-info now reflect the custom model.
    assert client.get("/models/active").json()["model_id"] == model_id
    assert client.get(f"/models/{model_id}").json()["active"] is True

    # A prediction is served by the active model and still emits a standardized event.
    pred = client.post("/predict", json={"income": 4200.0, "request_id": "act-1"})
    assert pred.status_code == 200
    assert pred.json()["event_emitted"] is True


@pytest.mark.integration
def test_deactivate_restores_default_model(client) -> None:
    files = {"file": ("model.pkl", _model_bytes(), "application/octet-stream")}
    model_id = client.post("/models/upload", files=files).json()["model_id"]
    client.post(f"/models/{model_id}/activate")

    resp = client.post("/models/deactivate")
    assert resp.status_code == 200
    assert resp.json()["is_custom"] is False
    assert client.get("/models/active").json()["model_id"] is None


@pytest.mark.integration
def test_activate_unknown_model_404(client) -> None:
    assert client.post("/models/does-not-exist/activate").status_code == 404
