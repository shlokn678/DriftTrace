"""API smoke test for DriftTrace (model-agnostic, local-first, no Docker, no broker).

Exercises the real serving app in-process via FastAPI's TestClient with NO built-in
model, building bundles on the fly:

    fresh app        -> no active model, /predict is 503
    classifier+graph -> upload -> inspect -> activate -> predict -> drift test
                        -> the earliest drifted feature is the root cause
    regression+graph -> activate -> predict returns a numeric output
    classifier no-graph -> drift test detects drift but no root-cause tracing

Run:  python scripts/smoke_api.py
Exits non-zero with a clear message on the first failed check.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile


def _check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"  ok: {label}", flush=True)


def _bundle(model, reference, edges=None) -> bytes:
    import cloudpickle

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        mb = io.BytesIO()
        cloudpickle.dump(model, mb)
        zf.writestr("model.pkl", mb.getvalue())
        zf.writestr("reference.csv", reference.to_csv(index=False))
        if edges is not None:
            zf.writestr("graph.json", json.dumps({"edges": edges}))
    return buf.getvalue()


def main() -> int:
    try:
        import numpy as np
        import pandas as pd
        from fastapi.testclient import TestClient
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression

        from drifttrace.serving.app import create_app
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] ERROR: could not import dependencies: {exc}", file=sys.stderr)
        print('[smoke] install: pip install -e ".[serving]"', file=sys.stderr)
        return 1

    rng = np.random.default_rng(0)
    clf_ref = pd.DataFrame(
        {
            "feature_a": rng.normal(40, 10, 400),
            "feature_b": rng.normal(50000, 12000, 400),
            "feature_c": rng.normal(5, 2, 400),
        }
    )
    clf_y = (clf_ref["feature_b"] + clf_ref["feature_a"] * 100 > 55000).astype(int)
    clf = LogisticRegression(max_iter=200).fit(clf_ref, clf_y)
    clf_edges = [["feature_a", "feature_b"], ["feature_b", "feature_c"]]

    reg_ref = pd.DataFrame(
        {
            "temperature": rng.normal(20, 5, 400),
            "humidity": rng.normal(60, 15, 400),
        }
    )
    reg_y = reg_ref["temperature"] * 2 + rng.normal(0, 1, 400)
    reg = GradientBoostingRegressor(random_state=0).fit(reg_ref, reg_y)

    def upload(client, data):
        return client.post("/models/upload", files={"file": ("b.zip", data, "application/zip")})

    print("[smoke] starting DriftTrace app (in-process) ...", flush=True)
    with TestClient(create_app()) as client:
        _check(client.get("/health").json()["status"] == "ok", "/health is ok")

        # Fresh: no active model.
        _check(client.get("/ready").json()["ready"] is False, "fresh app has no active model")
        _check(client.post("/predict", json={"features": {"x": 1}}).status_code == 503,
               "/predict is 503 with no active model")

        # Classifier + graph.
        up = upload(client, _bundle(clf, clf_ref, clf_edges)).json()
        _check(up["supported"] and up["task"] == "classification", "classifier inspected")
        _check(up["dependencies_available"] is True, "graph detected")
        cid = up["model_id"]
        _check(client.post(f"/models/{cid}/activate").json()["active"] is True, "classifier activated")
        _check(client.get("/model-info").json()["features"] == ["feature_a", "feature_b", "feature_c"],
               "/model-info shows dynamic features")
        pred = client.post(
            "/predict",
            json={"features": {"feature_a": 40, "feature_b": 60000, "feature_c": 5}},
        ).json()
        _check(pred["prediction"] in (0, 1) and pred["event_emitted"] is True, "prediction + event")

        dt = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
        oc = dt["outcome"]
        _check(oc is not None and len(oc["drifted_nodes"]) > 0, "drift test detected drift")
        _check("feature_a" in oc["root_cause_candidates"], "feature_a is the root cause (graph RCA)")
        _check(client.get("/rca/latest").json()["available"] is True, "/rca/latest persisted")

        # Regression + graph.
        rid = upload(client, _bundle(reg, reg_ref, [["temperature", "humidity"]])).json()["model_id"]
        client.post(f"/models/{rid}/activate")
        rpred = client.post(
            "/predict", json={"features": {"temperature": 21, "humidity": 55}}
        ).json()
        _check(rpred["prediction"] is None and rpred["output"] is not None, "regression returns numeric output")

        # Classifier without a graph.
        ng = upload(client, _bundle(clf, clf_ref, None)).json()
        _check(ng["dependencies_available"] is False, "no-graph model reports no dependencies")
        client.post(f"/models/{ng['model_id']}/activate")
        ngdt = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
        _check(ngdt["dependencies_available"] is False, "no-graph drift test has no dependency tracing")
        _check(ngdt["outcome"]["symptoms"] == [], "no-graph: no downstream symptoms")

    print("[smoke] all checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n[smoke] FAILED: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc
