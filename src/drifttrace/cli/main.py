"""DriftTrace command-line interface (FR-6.4, FR-12.4).

Thin wrapper over library functions so the whole pipeline runs without Airflow or a
broker. Commands:

- ``build-dataset``  generate + validate + persist the dataset
- ``run-pipeline``   train, evaluate, build baseline, log/register
- ``ingest``         DAG stage: ingest (generate/persist dataset)
- ``validate``       DAG stage: validate against schema (blocks on failure)
- ``drift-check``    DAG stage: per-node drift vs baseline over a data window
- ``report``         DAG stage: write the pipeline report (always runs)
- ``retrain``        DAG stage: retrain, gated by explicit --approve (FR-6.2/FR-12.2)
- ``run-dag``        run the whole ingest->validate->drift-check->report[/retrain] locally

Later phases add ``serve``, ``monitor``, ``rca show``, ``rollback``, ``inject``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from drifttrace.data.build import build_dataset
from drifttrace.orchestration.stages import (
    ApprovalRequired,
    ValidationFailed,
    drift_check_stage,
    ingest_stage,
    report_stage,
    retrain_stage,
    validate_stage,
)
from drifttrace.training.pipeline import run_training_pipeline
from drifttrace.training.train import TrainConfig


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


# ---- Phase 1 commands -----------------------------------------------------------------
def _manifest_summary(manifest: dict) -> dict:
    return {
        "dataset_path": manifest["dataset_path"],
        "dataset_version": manifest["dataset_version"],
        "n_rows": manifest["n_rows"],
        "validation_passed": manifest["validation"]["passed"],
    }


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    manifest = build_dataset(n_rows=args.n_rows, seed=args.seed)
    _emit({"stage": "build-dataset", **_manifest_summary(manifest)})
    return 0


def _cmd_run_pipeline(args: argparse.Namespace) -> int:
    config = TrainConfig(model=args.model, seed=args.seed, min_roc_auc=args.min_roc_auc)
    result = run_training_pipeline(config=config)
    _emit({"stage": "run-pipeline", **result.to_dict()})
    return 0


# ---- Phase 2 DAG-stage commands -------------------------------------------------------
def _cmd_ingest(args: argparse.Namespace) -> int:
    result = ingest_stage(n_rows=args.n_rows, seed=args.seed)
    _emit(result.to_dict())
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        result = validate_stage()
    except ValidationFailed as exc:
        _emit({"stage": "validate", "ok": False, "detail": json.loads(str(exc))})
        return 1
    _emit(result.to_dict())
    return 0


def _cmd_drift_check(args: argparse.Namespace) -> int:
    ds = Path(args.data) if args.data else None
    frame = pd.read_csv(ds) if ds else pd.read_csv(_default_dataset())
    result = drift_check_stage(frame, rel_threshold=args.threshold)
    _emit(result.to_dict())
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    ds = Path(args.data) if args.data else None
    frame = pd.read_csv(ds) if ds else pd.read_csv(_default_dataset())
    drift = drift_check_stage(frame, rel_threshold=args.threshold)
    result = report_stage(drift)
    _emit(result.to_dict())
    return 0


def _cmd_retrain(args: argparse.Namespace) -> int:
    try:
        result = retrain_stage(approved=args.approve, approver=args.approver)
    except ApprovalRequired as exc:
        _emit({"stage": "retrain", "ok": False, "detail": {"error": str(exc)}})
        return 2
    _emit(result.to_dict())
    return 0


def _cmd_run_dag(args: argparse.Namespace) -> int:
    """Run the full local DAG sequence (FR-6 AC-4: same artifacts as Airflow)."""
    steps: list[dict] = []
    steps.append(ingest_stage(n_rows=args.n_rows, seed=args.seed).to_dict())
    try:
        steps.append(validate_stage().to_dict())
    except ValidationFailed as exc:
        steps.append({"stage": "validate", "ok": False, "detail": json.loads(str(exc))})
        _emit({"dag": "drifttrace", "ok": False, "steps": steps})
        return 1

    frame = pd.read_csv(_default_dataset())
    drift = drift_check_stage(frame, rel_threshold=args.threshold)
    steps.append(drift.to_dict())
    steps.append(report_stage(drift).to_dict())  # report always runs

    if args.approve:
        steps.append(retrain_stage(approved=True, approver=args.approver).to_dict())
    else:
        steps.append({"stage": "retrain", "ok": True, "detail": {"skipped": "no approval (gated)"}})
    _emit({"dag": "drifttrace", "ok": True, "steps": steps})
    return 0


def _default_dataset() -> Path:
    from drifttrace.config import get_paths

    return get_paths().data / "dataset.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drifttrace", description="DriftTrace CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-dataset", help="generate + validate + persist the dataset")
    p_build.add_argument("--n-rows", type=int, default=5000)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.set_defaults(func=_cmd_build_dataset)

    p_run = sub.add_parser("run-pipeline", help="train, evaluate, build baseline, log/register")
    p_run.add_argument("--model", default="gradient_boosting")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--min-roc-auc", type=float, default=0.7)
    p_run.set_defaults(func=_cmd_run_pipeline)

    p_ingest = sub.add_parser("ingest", help="DAG stage: ingest the dataset")
    p_ingest.add_argument("--n-rows", type=int, default=5000)
    p_ingest.add_argument("--seed", type=int, default=42)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_validate = sub.add_parser("validate", help="DAG stage: validate against schema")
    p_validate.set_defaults(func=_cmd_validate)

    p_drift = sub.add_parser("drift-check", help="DAG stage: per-node drift vs baseline")
    p_drift.add_argument("--data", default=None, help="CSV window; defaults to data/dataset.csv")
    p_drift.add_argument("--threshold", type=float, default=0.2)
    p_drift.set_defaults(func=_cmd_drift_check)

    p_report = sub.add_parser("report", help="DAG stage: write the pipeline report")
    p_report.add_argument("--data", default=None)
    p_report.add_argument("--threshold", type=float, default=0.2)
    p_report.set_defaults(func=_cmd_report)

    p_retrain = sub.add_parser("retrain", help="DAG stage: retrain (requires --approve)")
    p_retrain.add_argument("--approve", action="store_true", help="explicit human approval")
    p_retrain.add_argument("--approver", default=None, help="who approved the retrain")
    p_retrain.set_defaults(func=_cmd_retrain)

    p_dag = sub.add_parser("run-dag", help="run the full DAG sequence locally")
    p_dag.add_argument("--n-rows", type=int, default=5000)
    p_dag.add_argument("--seed", type=int, default=42)
    p_dag.add_argument("--threshold", type=float, default=0.2)
    p_dag.add_argument("--approve", action="store_true")
    p_dag.add_argument("--approver", default=None)
    p_dag.set_defaults(func=_cmd_run_dag)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
