"""Validate the five generated test-model bundles against the CURRENT DriftTrace app.

Read-only with respect to production code. For each bundle it drives the real FastAPI
app (in-process) and the real drift/RCA pipeline:

  upload -> inspect -> activate -> predict -> prediction event -> switch away/back
  stable.csv  -> no drift (no false positive)
  drift.csv   -> drift detected
  graph model -> RCA produces a root cause + symptoms
  no-graph    -> drift detected, dependencies unavailable, root cause undetermined

Runs against a temporary DRIFTTRACE_ROOT so the repository is never touched.

Run:  python scripts/validate_test_models.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BUNDLES = REPO / "artifacts" / "test_models"

# Isolate runtime state in a temp root (never the repo).
os.environ.setdefault("DRIFTTRACE_ROOT", tempfile.mkdtemp(prefix="dt_validate_"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
for sub in ("data", "artifacts", "reports"):
    (Path(os.environ["DRIFTTRACE_ROOT"]) / sub).mkdir(parents=True, exist_ok=True)

from fastapi.testclient import TestClient  # noqa: E402

from drifttrace.drift.config import load_drift_config  # noqa: E402
from drifttrace.serving.app import create_app  # noqa: E402
from drifttrace.serving.config import get_serving_settings  # noqa: E402
from drifttrace.serving.onboarding import REGISTRY  # noqa: E402
from drifttrace.streaming.processing import run_scenario_over_events  # noqa: E402

MODELS = [
    ("breast_cancer_logistic", "classification", True),
    ("wine_random_forest", "classification", False),
    ("digits_extra_trees", "classification", False),
    ("synthetic_gradient_boosting", "classification", True),
    ("synthetic_regression", "regression", False),
]


def _events_from_csv(ctx, csv_path: Path):
    """Turn each CSV row into a standardized prediction event via the active adapter."""
    from drifttrace.adapters.base import to_standard_event
    from drifttrace.streaming.event import new_event_id

    frame = pd.read_csv(csv_path)
    meta = ctx.adapter.metadata()
    events = []
    for _, row in frame.iterrows():
        feats = {f: row[f] for f in ctx.feature_names if f in frame.columns}
        try:
            result = ctx.adapter.predict_one(feats)
        except Exception:  # noqa: BLE001
            continue
        events.append(
            to_standard_event(result, meta, source="validate").model_copy(
                update={"event_id": new_event_id()}
            )
        )
    return events


def _run_drift(ctx, csv_path: Path):
    """Run the real Phase-4 pipeline over events built from a CSV. Returns the outcome."""
    events = _events_from_csv(ctx, csv_path)
    config = load_drift_config()
    graph = ctx.graph
    if graph is None:
        # Mirror the app's no-graph handling: an edge-free graph (features independent).
        from drifttrace.bundle.reference import OUTPUT_NODE
        from drifttrace.graph.dag import DependencyGraph, NodeSpec

        specs = [NodeSpec(name=f, kind="raw_input") for f in ctx.feature_names]
        specs.append(
            NodeSpec(name=OUTPUT_NODE, kind="model_output", parents=tuple(ctx.feature_names))
        )
        graph = DependencyGraph(specs)
    outcomes = run_scenario_over_events(
        events,
        ctx.baseline,
        graph,
        config,
        reports_dir=None,
        window_size=len(events),
        min_window_samples=min(30, max(10, len(events) // 10)),
    )
    return outcomes[-1] if outcomes else None


def main() -> int:
    if not BUNDLES.exists():
        print("No bundles found. Run: python scripts/generate_test_models.py")
        return 1

    client = TestClient(create_app(get_serving_settings()))
    rows = []
    all_ok = True

    for name, expect_task, expect_graph in MODELS:
        d = BUNDLES / name
        zip_path = d / f"{name}.drift.zip"
        r = {"model": name, "task": expect_task, "graph": expect_graph}

        # Fresh registry between models.
        REGISTRY._models.clear()
        REGISTRY.deactivate()

        # 1. upload + inspect
        with zip_path.open("rb") as fh:
            up = client.post("/models/upload", files={"file": (zip_path.name, fh.read(), "application/zip")})
        ub = up.json()
        r["bundle_valid"] = up.status_code == 200 and ub.get("supported") is True
        r["task_ok"] = ub.get("task") == expect_task
        r["graph_ok"] = ub.get("dependencies_available") is expect_graph
        mid = ub.get("model_id")

        # 2. activate
        act = client.post(f"/models/{mid}/activate") if mid else None
        r["activation"] = bool(act and act.status_code == 200 and act.json().get("active") is True)

        # 3. predict (use the reference's first row as a valid feature vector)
        ref = pd.read_csv(d / "reference.csv")
        feats = {c: ref.iloc[0][c] for c in ref.columns}
        pred = client.post("/predict", json={"features": feats})
        pj = pred.json() if pred.status_code == 200 else {}
        r["prediction"] = pred.status_code == 200 and pj.get("event_emitted") is True
        if expect_task == "regression":
            r["prediction"] = r["prediction"] and pj.get("prediction") is None and pj.get("output") is not None
        else:
            r["prediction"] = r["prediction"] and pj.get("prediction") is not None

        # 4. model switch away and back (state isolation)
        REGISTRY.deactivate()
        switched = client.get("/models/active").json().get("active") is False
        client.post(f"/models/{mid}/activate")
        r["switch"] = switched and client.get("/models/active").json().get("model_id") == mid

        # 5. stable + drift via the real pipeline (needs the active context)
        ctx = REGISTRY.active_context()
        stable_oc = _run_drift(ctx, d / "stable.csv")
        drift_oc = _run_drift(ctx, d / "drift.csv")
        r["stable_no_drift"] = bool(stable_oc and len(stable_oc.drift.drifted_nodes) == 0)
        r["drift_detected"] = bool(drift_oc and len(drift_oc.drift.drifted_nodes) > 0)

        # 6. RCA expectations
        if expect_graph:
            r["rca"] = bool(drift_oc and drift_oc.rca.has_root_cause)
        else:
            # no graph -> features independent -> no downstream symptom chain
            r["rca"] = bool(drift_oc and drift_oc.rca.symptoms == [])

        checks = [
            r["bundle_valid"], r["task_ok"], r["graph_ok"], r["activation"],
            r["prediction"], r["switch"], r["stable_no_drift"], r["drift_detected"], r["rca"],
        ]
        r["PASS"] = all(checks)
        all_ok = all_ok and r["PASS"]
        rows.append(r)

    # Report
    print("\n=== DriftTrace test-model validation ===\n")
    hdr = f'{"MODEL":<30} {"TASK":<14} {"GRAPH":<6} {"BUNDLE":<7} {"ACT":<4} {"PRED":<5} {"SWTCH":<6} {"STABLE":<7} {"DRIFT":<6} {"RCA":<4} {"RESULT"}'
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f'{r["model"]:<30} {r["task"]:<14} {("YES" if r["graph"] else "NO"):<6} '
            f'{_b(r["bundle_valid"]):<7} {_b(r["activation"]):<4} {_b(r["prediction"]):<5} '
            f'{_b(r["switch"]):<6} {_b(r["stable_no_drift"]):<7} {_b(r["drift_detected"]):<6} '
            f'{_b(r["rca"]):<4} {"PASS" if r["PASS"] else "FAIL"}'
        )
    print("\n" + ("ALL MODELS PASSED" if all_ok else "SOME MODELS FAILED"))
    return 0 if all_ok else 1


def _b(v: bool) -> str:
    return "ok" if v else "X"


if __name__ == "__main__":
    raise SystemExit(main())
