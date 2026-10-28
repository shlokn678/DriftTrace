"""Validate the DVC-tracked demo data and write a metrics summary.

This script is the single stage of the minimal DVC pipeline (see ``dvc.yaml``).
It reads the DVC-tracked copies under ``mlops_demo_data/`` (NOT the originals in
``artifacts/test_models/``) and checks, for each of the five models, that:

  * reference.csv, stable.csv, drift.csv and metrics.json all exist, and
  * the three CSVs share the same set of feature columns.

It records row counts and the feature count, pulls the pre-recorded scalar
metrics out of metrics.json, and writes everything to ``mlops_demo_summary.json``
at the repo root (the DVC metrics output).

Usage (from the repo root, with the venv active):

    python scripts/validate_mlops_demo_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Resolve the repo root from this script's location (scripts/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = REPO_ROOT / "mlops_demo_data"
SUMMARY_PATH = REPO_ROOT / "mlops_demo_summary.json"

# The five models, in the canonical order.
MODELS = [
    "breast_cancer_logistic",
    "wine_random_forest",
    "digits_extra_trees",
    "synthetic_gradient_boosting",
    "synthetic_regression",
]

CSV_FILES = ("reference.csv", "stable.csv", "drift.csv")


def _is_scalar_number(value) -> bool:
    """True only for real numeric scalars (excludes bools, lists, dicts, None)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_metrics(metrics: dict) -> dict:
    """Keep only numeric scalar metrics (drops confusion_matrix / lists / None)."""
    return {k: v for k, v in metrics.items() if _is_scalar_number(v)}


def validate_model(model_name: str) -> dict:
    """Validate one model's demo data and return its summary record."""
    model_dir = DEMO_DATA_DIR / model_name
    record: dict = {
        "model": model_name,
        "status": "fail",
        "reason": "",
        "row_counts": {},
        "feature_count": None,
        "columns_match": False,
        "task": None,
        "algorithm": None,
        "metrics": {},
    }

    # Check required files exist.
    missing = [
        name
        for name in (*CSV_FILES, "metrics.json")
        if not (model_dir / name).is_file()
    ]
    if missing:
        record["reason"] = f"missing files: {', '.join(missing)}"
        return record

    # Load the three CSVs and compare column sets.
    column_sets: list[set[str]] = []
    first_columns: list[str] | None = None
    for name in CSV_FILES:
        df = pd.read_csv(model_dir / name)
        record["row_counts"][name] = int(len(df))
        column_sets.append(set(df.columns))
        if first_columns is None:
            first_columns = list(df.columns)

    columns_match = all(cols == column_sets[0] for cols in column_sets)
    record["columns_match"] = bool(columns_match)
    record["feature_count"] = len(first_columns or [])

    # Read metrics.json for the scalar metrics and descriptive fields.
    with (model_dir / "metrics.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    record["task"] = meta.get("task")
    record["algorithm"] = meta.get("algorithm")
    record["metrics"] = _scalar_metrics(meta.get("metrics") or {})

    if not columns_match:
        record["reason"] = "CSV feature columns do not match across reference/stable/drift"
        return record

    record["status"] = "pass"
    return record


def main() -> int:
    models_summary = []
    for model_name in MODELS:
        print(f"Validating: {model_name}")
        rec = validate_model(model_name)
        status = rec["status"].upper()
        detail = f" ({rec['reason']})" if rec["reason"] else ""
        print(f"  [{status}]{detail} rows={rec['row_counts']} features={rec['feature_count']}")
        models_summary.append(rec)

    all_valid = all(rec["status"] == "pass" for rec in models_summary)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models": len(models_summary),
        "all_valid": all_valid,
        "models": models_summary,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nWrote summary: {SUMMARY_PATH}")
    print(f"all_valid = {all_valid} ({len(models_summary)} models)")

    # Exit non-zero if any model failed so the DVC stage surfaces the failure.
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
