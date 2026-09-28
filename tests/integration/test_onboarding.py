"""Integration tests for the model-agnostic bundle onboarding + active-model workflow.

Builds real (small) scikit-learn bundles - a classifier with a graph, a classifier
without a graph, and a regression Pipeline - and exercises upload -> inspect -> activate
-> predict -> drift-test through the actual FastAPI app. No loan domain, no built-in
model: a fresh app starts with NO active model.
"""

from __future__ import annotations

import io
import json
import zipfile

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient over a temp project root with no model active."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app
    from drifttrace.serving.config import get_serving_settings
    from drifttrace.serving.onboarding import REGISTRY

    # Reset the process-wide registry between tests.
    REGISTRY._models.clear()
    REGISTRY.deactivate()
    return TestClient(create_app(get_serving_settings()))


# ---- bundle builders ------------------------------------------------------------------
def _zip_bundle(model, reference: pd.DataFrame, edges=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        mb = io.BytesIO()
        cloudpickle.dump(model, mb)
        zf.writestr("model.pkl", mb.getvalue())
        zf.writestr("reference.csv", reference.to_csv(index=False))
        if edges is not None:
            zf.writestr("graph.json", json.dumps({"edges": edges}))
    return buf.getvalue()


def _classifier_with_graph() -> bytes:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(40, 10, n),
            "feature_b": rng.normal(50000, 12000, n),
            "feature_c": rng.normal(5, 2, n),
        }
    )
    y = (df["feature_b"] + df["feature_a"] * 100 > 55000).astype(int)
    model = LogisticRegression(max_iter=200).fit(df, y)
    return _zip_bundle(model, df, edges=[["feature_a", "feature_b"], ["feature_b", "feature_c"]])


def _classifier_no_graph() -> bytes:
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "transaction_amount": rng.normal(200, 60, n),
            "merchant_score": rng.normal(0.5, 0.2, n),
            "device_score": rng.normal(0.5, 0.2, n),
            "country_score": rng.normal(0.5, 0.2, n),
        }
    )
    y = (df["transaction_amount"] > 200).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(df, y)
    return _zip_bundle(model, df, edges=None)


def _regression_pipeline() -> bytes:
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame(
        {
            "temperature": rng.normal(20, 5, n),
            "humidity": rng.normal(60, 15, n),
            "pressure": rng.normal(1013, 8, n),
            "wind_speed": rng.normal(10, 4, n),
        }
    )
    y = df["temperature"] * 2 + df["humidity"] * 0.1 + rng.normal(0, 1, n)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("gbr", GradientBoostingRegressor(random_state=0))]
    ).fit(df, y)
    return _zip_bundle(pipe, df, edges=[["temperature", "humidity"], ["humidity", "wind_speed"]])


def _upload(client, data: bytes):
    return client.post("/models/upload", files={"file": ("bundle.zip", data, "application/zip")})


# ---- fresh state ----------------------------------------------------------------------
@pytest.mark.integration
def test_fresh_state_has_no_active_model(client) -> None:
    assert client.get("/ready").json()["ready"] is False
    active = client.get("/models/active").json()
    assert active["active"] is False
    assert active["model_id"] is None
    # No model -> predict is unavailable.
    assert client.post("/predict", json={"features": {"x": 1.0}}).status_code == 503


# ---- upload + inspection --------------------------------------------------------------
@pytest.mark.integration
def test_upload_classifier_with_graph(client) -> None:
    body = _upload(client, _classifier_with_graph()).json()
    assert body["supported"] is True
    assert body["framework"] == "scikit-learn"
    assert body["task"] == "classification"
    assert body["n_features"] == 3
    assert body["features"] == ["feature_a", "feature_b", "feature_c"]
    assert body["reference_available"] is True
    assert body["dependencies_available"] is True
    assert body["ready_to_monitor"] is True
    assert body["missing"] == []
    assert body["model_id"]


@pytest.mark.integration
def test_upload_classifier_no_graph_reports_missing_dependencies(client) -> None:
    body = _upload(client, _classifier_no_graph()).json()
    assert body["supported"] is True
    assert body["dependencies_available"] is False
    assert "dependencies" in body["missing"]
    # Still monitorable: reference data exists.
    assert body["ready_to_monitor"] is True


@pytest.mark.integration
def test_upload_regression_pipeline_detected(client) -> None:
    body = _upload(client, _regression_pipeline()).json()
    assert body["supported"] is True
    assert body["task"] == "regression"
    assert body["n_features"] == 4


@pytest.mark.integration
def test_upload_unsupported_file(client) -> None:
    resp = _upload(client, b"not a zip and not a model")
    # A non-zip, non-model upload is rejected with a clear 4xx.
    assert resp.status_code == 422


@pytest.mark.integration
def test_upload_requires_file(client) -> None:
    assert client.post("/models/upload").status_code == 422


# ---- activation + prediction ----------------------------------------------------------
@pytest.mark.integration
def test_activate_classifier_and_predict(client) -> None:
    mid = _upload(client, _classifier_with_graph()).json()["model_id"]
    active = client.post(f"/models/{mid}/activate").json()
    assert active["active"] is True
    assert active["model_id"] == mid
    assert active["task"] == "classification"
    assert active["dependencies_available"] is True

    assert client.get("/ready").json()["ready"] is True
    info = client.get("/model-info").json()
    assert info["features"] == ["feature_a", "feature_b", "feature_c"]

    pred = client.post(
        "/predict",
        json={"features": {"feature_a": 40, "feature_b": 60000, "feature_c": 5}},
    )
    assert pred.status_code == 200
    body = pred.json()
    assert body["prediction"] in (0, 1)
    assert body["event_emitted"] is True
    assert set(body["features"]) == {"feature_a", "feature_b", "feature_c"}


@pytest.mark.integration
def test_activate_regression_pipeline_and_predict(client) -> None:
    mid = _upload(client, _regression_pipeline()).json()["model_id"]
    client.post(f"/models/{mid}/activate")
    pred = client.post(
        "/predict",
        json={
            "features": {
                "temperature": 21,
                "humidity": 55,
                "pressure": 1010,
                "wind_speed": 8,
            }
        },
    ).json()
    # Regressors report a numeric output and no discrete class.
    assert pred["prediction"] is None
    assert pred["output"] is not None
    assert pred["event_emitted"] is True


@pytest.mark.integration
def test_activate_unknown_model_404(client) -> None:
    assert client.post("/models/does-not-exist/activate").status_code == 404


# ---- drift test + RCA -----------------------------------------------------------------
@pytest.mark.integration
def test_drift_test_with_graph_traces_root_cause(client) -> None:
    mid = _upload(client, _classifier_with_graph()).json()["model_id"]
    client.post(f"/models/{mid}/activate")
    res = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
    assert res["dependencies_available"] is True
    outcome = res["outcome"]
    assert outcome is not None
    assert len(outcome["drifted_nodes"]) > 0
    # With the a->b->c graph and a broad shift, feature_a is the upstream root cause.
    assert outcome["has_root_cause"] is True
    assert "feature_a" in outcome["root_cause_candidates"]
    # /rca/latest reflects the same outcome.
    rca = client.get("/rca/latest").json()
    assert rca["available"] is True
    assert rca["dependencies_available"] is True


@pytest.mark.integration
def test_drift_test_no_graph_has_no_root_cause_tracing(client) -> None:
    mid = _upload(client, _classifier_no_graph()).json()["model_id"]
    client.post(f"/models/{mid}/activate")
    res = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
    assert res["dependencies_available"] is False
    outcome = res["outcome"]
    assert outcome is not None
    assert len(outcome["drifted_nodes"]) > 0
    # No graph -> features are independent -> no downstream symptom chain.
    assert outcome["symptoms"] == []


@pytest.mark.integration
def test_baseline_drift_test_reports_no_drift(client) -> None:
    mid = _upload(client, _classifier_with_graph()).json()["model_id"]
    client.post(f"/models/{mid}/activate")
    # intensity 0 = no perturbation -> the sample matches the reference.
    res = client.post("/demo/run-drift-test", json={"intensity": 0.0, "n": 200, "seed": 7}).json()
    outcome = res["outcome"]
    assert outcome is not None
    assert outcome["has_root_cause"] is False


# ---- model switching ------------------------------------------------------------------
@pytest.mark.integration
def test_model_switching_resets_state(client) -> None:
    mid_a = _upload(client, _classifier_with_graph()).json()["model_id"]
    client.post(f"/models/{mid_a}/activate")
    client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7})
    assert client.get("/rca/latest").json()["available"] is True

    # Switch to a different model with different features.
    mid_b = _upload(client, _classifier_no_graph()).json()["model_id"]
    active = client.post(f"/models/{mid_b}/activate").json()
    assert active["model_id"] == mid_b
    assert active["features"] == [
        "transaction_amount",
        "merchant_score",
        "device_score",
        "country_score",
    ]
    # Switching clears the previous model's report (no stale mixing).
    assert client.get("/rca/latest").json()["available"] is False
    # model-info reflects the new feature set.
    assert client.get("/model-info").json()["features"] == active["features"]


@pytest.mark.integration
def test_deactivate_stops_serving(client) -> None:
    mid = _upload(client, _classifier_with_graph()).json()["model_id"]
    client.post(f"/models/{mid}/activate")
    assert client.get("/ready").json()["ready"] is True
    client.post("/models/deactivate")
    assert client.get("/ready").json()["ready"] is False
    assert client.get("/models/active").json()["active"] is False
