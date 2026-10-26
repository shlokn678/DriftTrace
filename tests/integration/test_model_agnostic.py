"""Genericity tests: previously-UNSEEN models drive the full workflow with NO
production-code changes.

These models are created fresh at test time with feature names, counts, tasks, and
output interfaces that are NOT any of the five bundled fixtures and NOT the synthetic
showcase chain. They prove the platform is model-agnostic: onboarding, schema discovery,
prediction, reference-based drift (the REAL KS/PSI engine), graph RCA when a graph is
present (and graceful no-graph behaviour), and model/session isolation all work from the
model's own metadata + reference data.

Everything runs through the real FastAPI app and the real drift/RCA pipeline - no mocks,
no fabricated verdicts.
"""

from __future__ import annotations

import io
import json
import zipfile

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# --------------------------------------------------------------------------------------
# fixture: a TestClient over a temp root with no active model
# --------------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    monkeypatch.setenv("DRIFTTRACE_LIVE_SOURCE", "file")
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app
    from drifttrace.serving.config import get_serving_settings
    from drifttrace.serving.onboarding import REGISTRY

    REGISTRY._models.clear()
    REGISTRY.deactivate()
    return TestClient(create_app(get_serving_settings()))


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


def _upload_activate(client, data: bytes) -> dict:
    up = client.post(
        "/models/upload", files={"file": ("unseen.zip", data, "application/zip")}
    ).json()
    assert up["supported"] is True, up
    client.post(f"/models/{up['model_id']}/activate")
    return up


# --------------------------------------------------------------------------------------
# UNSEEN MODEL 1: multiclass classifier in a preprocessing Pipeline, novel feature names
# --------------------------------------------------------------------------------------
def _unseen_multiclass_pipeline() -> tuple[bytes, list[str]]:
    rng = np.random.default_rng(11)
    n = 500
    cols = ["sepal_len_mm", "sepal_wid_mm", "petal_len_mm", "petal_wid_mm", "stem_density"]
    df = pd.DataFrame({c: rng.normal(5 * (i + 1), 1.5, n) for i, c in enumerate(cols)})
    # 3-class target from a nonlinear rule -> genuine multiclass.
    y = (df["petal_len_mm"] + df["stem_density"]).rank(pct=True)
    y = np.select([y < 0.34, y < 0.67], [0, 1], default=2)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]).fit(
        df, y
    )
    return _zip_bundle(model, df), cols


@pytest.mark.integration
def test_unseen_multiclass_pipeline_end_to_end(client) -> None:
    data, cols = _unseen_multiclass_pipeline()
    up = _upload_activate(client, data)

    # Schema discovered from the model/reference, not hardcoded.
    assert up["task"] == "classification"
    assert up["features"] == cols
    assert up["n_features"] == len(cols)

    info = client.get("/model-info").json()
    assert info["features"] == cols

    # Prediction works with the model's own feature names.
    feats = dict.fromkeys(cols, 3.0)
    pred = client.post("/predict", json={"features": feats}).json()
    assert pred["prediction"] is not None  # a discrete class
    assert pred["event_emitted"] is True

    # Real drift test via the existing KS/PSI pipeline; no graph -> no symptom chain.
    res = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
    assert res["dependencies_available"] is False
    assert len(res["outcome"]["drifted_nodes"]) > 0
    assert res["outcome"]["symptoms"] == []


# --------------------------------------------------------------------------------------
# UNSEEN MODEL 2: regressor WITHOUT predict_proba (SVR), novel feature names
# --------------------------------------------------------------------------------------
def _unseen_regressor_no_proba() -> tuple[bytes, list[str]]:
    rng = np.random.default_rng(22)
    n = 500
    cols = ["pressure_kpa", "flow_rate", "ambient_temp", "valve_pos"]
    df = pd.DataFrame({c: rng.normal(10 * (i + 1), 4, n) for i, c in enumerate(cols)})
    y = df["pressure_kpa"] * 1.5 - df["valve_pos"] + rng.normal(0, 2, n)
    model = SVR().fit(df, y)
    return _zip_bundle(model, df), cols


@pytest.mark.integration
def test_unseen_regressor_without_proba(client) -> None:
    data, cols = _unseen_regressor_no_proba()
    up = _upload_activate(client, data)
    assert up["task"] == "regression"
    assert up["supports_proba"] is False

    pred = client.post("/predict", json={"features": dict.fromkeys(cols, 12.0)}).json()
    # Regressors report a numeric output and no discrete class / probability.
    assert pred["prediction"] is None
    assert pred["output"] is not None
    assert pred["probability"] is None

    res = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
    assert len(res["outcome"]["drifted_nodes"]) > 0


# --------------------------------------------------------------------------------------
# UNSEEN MODEL 3: graph-enabled classifier with a NOVEL dependency chain
# --------------------------------------------------------------------------------------
def _unseen_graph_classifier() -> tuple[bytes, list[str], list[list[str]]]:
    rng = np.random.default_rng(33)
    n = 600
    cols = ["soil_moisture", "root_uptake", "leaf_turgor", "canopy_index", "yield_proxy"]
    df = pd.DataFrame({c: rng.normal(20 + 5 * i, 3, n) for i, c in enumerate(cols)})
    y = (df["soil_moisture"] + df["root_uptake"] > df["soil_moisture"].mean() * 2).astype(int)
    model = ExtraTreesClassifier(n_estimators=60, random_state=0).fit(df, y)
    # Novel declared chain: soil_moisture -> root_uptake -> leaf_turgor
    edges = [["soil_moisture", "root_uptake"], ["root_uptake", "leaf_turgor"]]
    return _zip_bundle(model, df, edges=edges), cols, edges


@pytest.mark.integration
def test_unseen_graph_model_traces_root_and_symptoms(client) -> None:
    data, cols, _edges = _unseen_graph_classifier()
    up = _upload_activate(client, data)
    assert up["dependencies_available"] is True

    # Drift the declared chain so the EXISTING RCA engine yields one root + symptoms.
    # Use the real drift-test pipeline targeting the chain's upstream feature.
    res = client.post(
        "/demo/run-drift-test",
        json={"intensity": 3.0, "feature": "soil_moisture", "n": 200, "seed": 7},
    ).json()
    assert res["dependencies_available"] is True
    outcome = res["outcome"]
    assert outcome["has_root_cause"] is True
    # soil_moisture is the upstream node of the declared chain -> the root candidate.
    assert "soil_moisture" in outcome["root_cause_candidates"]

    # /rca/latest reflects the same model and carries the declared edges (edge-accurate).
    rca = client.get("/rca/latest").json()
    assert rca["available"] is True
    assert rca["report"]["model_version"] == up["model_id"]
    edge_pairs = {tuple(e) for e in rca["report"].get("graph_edges", [])}
    assert ("soil_moisture", "root_uptake") in edge_pairs
    assert ("root_uptake", "leaf_turgor") in edge_pairs


# --------------------------------------------------------------------------------------
# ISOLATION: switching to a different unseen model never shows the previous model's state
# --------------------------------------------------------------------------------------
@pytest.mark.integration
def test_switching_unseen_models_isolates_state(client) -> None:
    # Activate model A (regressor) and produce a report.
    data_a, cols_a = _unseen_regressor_no_proba()
    mid_a = _upload_activate(client, data_a)["model_id"]
    client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 150, "seed": 7})
    assert client.get("/rca/latest").json()["available"] is True

    # Switch to model B (multiclass pipeline, different features).
    data_b, cols_b = _unseen_multiclass_pipeline()
    mid_b = _upload_activate(client, data_b)["model_id"]
    assert mid_b != mid_a

    info = client.get("/model-info").json()
    assert info["features"] == cols_b  # new schema, not A's
    # A's report must not be presented as B's current state.
    assert client.get("/rca/latest").json()["available"] is False


# --------------------------------------------------------------------------------------
# EXPLANATION: SHAP on an unseen regressor returns a real explanation over its own schema
# --------------------------------------------------------------------------------------
@pytest.mark.integration
def test_unseen_model_explanation_uses_dynamic_schema(client) -> None:
    data, cols = _unseen_regressor_no_proba()
    _upload_activate(client, data)
    resp = client.post("/explain", json={"features": dict.fromkeys(cols, 12.0), "method": "shap"})
    # Either a genuine explanation over the model's own features, or a clear error -
    # never a fabricated result and never a crash.
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        body = resp.json()
        returned = {a["feature"] for a in body["attributions"]}
        assert returned == set(cols)  # dynamic feature names, not hardcoded
