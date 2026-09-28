"""API smoke test for DriftTrace (local-first, no Docker, no broker).

Exercises the real serving app in-process via FastAPI's TestClient and asserts the
core contract holds after a fresh bootstrap:

    /health, /ready, /model-info, /predict,
    /models/active (default loan model),
    /demo/run-scenario income_annual  -> income is the ROOT CAUSE,
                                          credit_score + risk_score are SYMPTOMS,
    /rca/latest -> the persisted report reflects that outcome.

Run after ``python -m drifttrace.bootstrap``:

    python scripts/smoke_api.py

Exits non-zero with a clear message on the first failed check.
"""

from __future__ import annotations

import sys


def _check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"  ok: {label}", flush=True)


def main() -> int:
    try:
        from fastapi.testclient import TestClient

        from drifttrace.serving.app import create_app
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] ERROR: could not import the serving app: {exc}", file=sys.stderr)
        print('[smoke] install serving extras: pip install -e ".[serving,tracking]"', file=sys.stderr)
        return 1

    print("[smoke] starting DriftTrace app (in-process) ...", flush=True)
    with TestClient(create_app()) as client:
        _check(client.get("/health").json()["status"] == "ok", "/health is ok")

        ready = client.get("/ready").json()
        _check(ready["ready"] is True, "/ready reports the model is loaded")

        info = client.get("/model-info").json()
        _check(info["loaded"] is True, "/model-info reports a loaded model")

        active = client.get("/models/active").json()
        _check(active["is_custom"] is False, "default model is the loan model (not custom)")
        _check(active["name"] == "drifttrace-loan-default", "active model is drifttrace-loan-default")

        pred = client.post("/predict", json={"income": 4200.0, "request_id": "smoke-1"})
        _check(pred.status_code == 200, "/predict returns 200")
        body = pred.json()
        _check(set(body["features"]) >= {"income", "credit_score", "risk_score"}, "prediction has chained features")

        scenario = client.post(
            "/demo/run-scenario", json={"scenario": "income_annual", "n": 300, "seed": 7}
        ).json()
        outcome = scenario["outcome"]
        _check(outcome is not None, "income_annual produced an outcome")
        _check("income" in outcome["root_cause_candidates"], "income is the ROOT CAUSE")
        _check("credit_score" in outcome["symptoms"], "credit_score is a SYMPTOM")
        _check("risk_score" in outcome["symptoms"], "risk_score is a SYMPTOM")

        rca = client.get("/rca/latest").json()
        _check(rca["available"] is True, "/rca/latest has a persisted report")
        roots = [c["node"] for c in rca["report"]["rca"]["root_cause_candidates"]]
        _check("income" in roots, "/rca/latest reports income as the root cause")

    print("[smoke] all checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n[smoke] FAILED: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc
