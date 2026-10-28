"""Log the five DriftTrace test-model bundles to MLflow for experiment tracking.

This is a small demonstration utility. It does NOT retrain anything and does NOT
re-evaluate any model. It simply reads each model's existing ``metrics.json``
(written when the bundles were created) and records those pre-computed numbers as
one MLflow run per model, so the five models can be browsed/compared in the MLflow
UI. It is deliberately separate from the live drift-detection / RCA engine.

Usage (from the repo root, with the venv active):

    python scripts/log_test_models_mlflow.py

The tracking store is a local SQLite file (``mlflow.db``) at the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

# Resolve the repo root from this script's location (scripts/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MODELS_DIR = REPO_ROOT / "artifacts" / "test_models"

EXPERIMENT_NAME = "DriftTrace-Test-Models"

# Absolute sqlite tracking URI (forward slashes) so it works from any cwd.
TRACKING_URI = "sqlite:///" + REPO_ROOT.joinpath("mlflow.db").as_posix()

# The five models, in the canonical order.
MODELS = [
    "breast_cancer_logistic",
    "wine_random_forest",
    "digits_extra_trees",
    "synthetic_gradient_boosting",
    "synthetic_regression",
]


def _is_scalar_number(value) -> bool:
    """True only for real numeric scalars (excludes bools, lists, dicts, None).

    This is what keeps ``confusion_matrix`` (a nested list) out of MLflow metrics.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def log_model_run(model_name: str) -> None:
    """Read one model's metrics.json and log a single MLflow run for it."""
    metrics_path = TEST_MODELS_DIR / model_name / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    with mlflow.start_run(run_name=model_name):
        # Params: descriptive, stringified for safety.
        mlflow.log_param("model", str(data.get("model", model_name)))
        mlflow.log_param("algorithm", str(data.get("algorithm")))
        mlflow.log_param("task", str(data.get("task")))
        mlflow.log_param("dataset", str(data.get("dataset")))
        mlflow.log_param("n_features", str(data.get("n_features")))
        mlflow.log_param("graph", str(data.get("graph")))

        # Metrics: only real numeric scalars (skips confusion_matrix / lists / None).
        metrics = data.get("metrics") or {}
        logged_metric_count = 0
        for key, value in metrics.items():
            if _is_scalar_number(value):
                mlflow.log_metric(key, float(value))
                logged_metric_count += 1

        # Tags make clear these are pre-recorded, not freshly evaluated.
        mlflow.set_tag("metrics_source", "precomputed_from_metrics_json")
        mlflow.set_tag(
            "note",
            "Metrics read from existing metrics.json; NOT re-evaluated by this script.",
        )

        # Attach the source metrics.json as an artifact for traceability.
        mlflow.log_artifact(str(metrics_path))

        print(f"  logged {logged_metric_count} scalar metric(s)")


def main() -> int:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results: list[tuple[str, bool, str]] = []
    for model_name in MODELS:
        print(f"Logging run for: {model_name}")
        try:
            log_model_run(model_name)
            results.append((model_name, True, ""))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the rest
            print(f"  FAILED: {exc}")
            results.append((model_name, False, str(exc)))

    # Summary.
    print("\n=== MLflow logging summary ===")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Tracking   : {TRACKING_URI}")
    ok_count = 0
    for model_name, ok, reason in results:
        status = "OK" if ok else "FAILED"
        if ok:
            ok_count += 1
        suffix = f" ({reason})" if reason else ""
        print(f"  [{status}] {model_name}{suffix}")
    print(f"{ok_count}/{len(results)} runs logged successfully.")

    # Exit non-zero only if everything failed.
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
