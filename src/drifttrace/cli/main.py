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
- ``serve``          run the FastAPI prediction service (uvicorn)
- ``webhook-stub``   run the local webhook stub receiver (uvicorn)
- ``demo-generate``  write deterministic normal + simulated-drift event files
- ``replay``         replay a JSON-lines event file through the monitor -> webhook

Later phases add ``rca show``, ``rollback``, ``explain``.
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
    # Record the approved retrain in the audit trail (FR-12, FR-15.3).
    from drifttrace.governance.audit import record_action

    record_action("retrain", approved=True, approver=args.approver, detail=result.detail)
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


# ---- Phase 3 serving / streaming commands ---------------------------------------------
def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "drifttrace.serving.app:create_app",
        host=args.host,
        port=args.port,
        factory=True,
    )
    return 0


def _cmd_webhook_stub(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "drifttrace.alerting.webhook_stub:create_app",
        host=args.host,
        port=args.port,
        factory=True,
    )
    return 0


def _cmd_demo_generate(args: argparse.Namespace) -> int:
    from drifttrace.config import get_paths
    from drifttrace.streaming.demo import write_demo_datasets

    out_dir = Path(args.out) if args.out else (get_paths().reports / "demo")
    paths = write_demo_datasets(out_dir, n=args.n, seed=args.seed)
    _emit({"stage": "demo-generate", "files": {k: str(v) for k, v in paths.items()}})
    return 0


def _build_processor(webhook_url: str | None) -> object:
    """Construct the Phase 4 processor (drift -> RCA -> report -> alert) from config."""
    from drifttrace.alerting.alerter import Alerter
    from drifttrace.alerting.webhook import WebhookClient
    from drifttrace.config import get_paths
    from drifttrace.drift.baseline import Baseline
    from drifttrace.drift.config import load_drift_config
    from drifttrace.graph.loader import load_graph
    from drifttrace.streaming.processing import Phase4Processor

    paths = get_paths()
    baseline = Baseline.load(paths.artifacts / "baseline.json")
    graph = load_graph()
    config = load_drift_config()
    webhook = WebhookClient(webhook_url) if webhook_url else None
    alerter = Alerter(
        webhook=webhook,
        cooldown_seconds=config.cooldown_seconds,
        alert_log_path=paths.reports / "alerts.jsonl",
    )
    return Phase4Processor(
        baseline=baseline,
        graph=graph,
        config=config,
        alerter=alerter,
        reports_dir=paths.reports,
    )


def _cmd_replay(args: argparse.Namespace) -> int:
    from drifttrace.streaming.monitor import Monitor
    from drifttrace.streaming.source import FileReplaySource
    from drifttrace.streaming.window import WindowPolicy

    source = FileReplaySource(args.file)
    processor = _build_processor(args.webhook_url)
    policy = WindowPolicy(size=args.window_size, min_samples=args.min_samples)
    monitor = Monitor(source=source, policy=policy, processor=processor)
    result = monitor.run()
    _emit({"stage": "replay", "file": args.file, **result.to_dict()})
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    """Run the Phase 4 monitor against Redpanda (bounded poll) or a file source."""
    from drifttrace.streaming.monitor import Monitor
    from drifttrace.streaming.source import FileReplaySource, RedpandaSource
    from drifttrace.streaming.window import WindowPolicy

    if args.file:
        source: object = FileReplaySource(args.file)
    else:
        source = RedpandaSource(brokers=args.brokers, topic=args.topic)
    processor = _build_processor(args.webhook_url)
    policy = WindowPolicy(size=args.window_size, min_samples=args.min_samples)
    monitor = Monitor(source=source, policy=policy, processor=processor)  # type: ignore[arg-type]
    result = monitor.run(limit=args.limit)
    _emit({"stage": "monitor", **result.to_dict()})
    return 0


# ---- Phase 4 operator + explainability commands ---------------------------------------
def _cmd_rollback(args: argparse.Namespace) -> int:
    from drifttrace.governance.operator import ApprovalRequired, rollback

    try:
        result = rollback(args.to_version, approved=args.approve, approver=args.approver)
    except ApprovalRequired as exc:
        _emit({"stage": "rollback", "ok": False, "detail": {"error": str(exc)}})
        return 2
    _emit({"stage": "rollback", **result.to_dict()})
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from drifttrace.drift.baseline import Baseline  # noqa: F401  (ensures pkg import ok)
    from drifttrace.explain.explainer import lime_explain_local, shap_explain_local
    from drifttrace.serving.app import ModelHolder, _explain_background
    from drifttrace.serving.config import get_serving_settings

    holder = ModelHolder(get_serving_settings())
    holder.load()
    if not holder.ready:
        _emit({"stage": "explain", "ok": False, "detail": {"error": "model not loaded"}})
        return 1
    bg = _explain_background(holder)
    if args.method == "lime":
        exp = lime_explain_local(
            holder.model,
            args.income,
            holder.transform_params,
            bg,
            model_version=holder.model_version,
        )
    else:
        exp = shap_explain_local(
            holder.model,
            args.income,
            holder.transform_params,
            model_version=holder.model_version,
            background=bg,
        )
    _emit({"stage": "explain", "ok": True, "explanation": exp.to_dict()})
    return 0


def _cmd_governance(args: argparse.Namespace) -> int:
    """Run fairness + privacy checks over the training dataset + prediction events."""
    import json

    import pandas as pd

    from drifttrace.config import get_paths
    from drifttrace.governance.config import load_governance_config
    from drifttrace.governance.fairness import evaluate_fairness
    from drifttrace.governance.privacy import check_records
    from drifttrace.serving.app import ModelHolder
    from drifttrace.serving.config import get_serving_settings

    paths = get_paths()
    gov = load_governance_config()
    holder = ModelHolder(get_serving_settings())
    holder.load()
    out: dict = {"stage": "governance"}

    if holder.ready and holder.transform_params is not None:
        frame = pd.read_csv(paths.data / "dataset.csv")
        fairness = evaluate_fairness(holder.model, frame, holder.transform_params, gov)
        out["fairness"] = fairness.to_dict()

    # Privacy check over any persisted prediction events.
    events_path = paths.reports / "events.jsonl"
    records = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    out["privacy"] = check_records(records, gov).to_dict()
    _emit(out)
    return 0


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

    p_serve = sub.add_parser("serve", help="run the FastAPI prediction service")
    p_serve.add_argument("--host", default="0.0.0.0")  # noqa: S104 - internal container bind
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_ws = sub.add_parser("webhook-stub", help="run the local webhook stub receiver")
    p_ws.add_argument("--host", default="0.0.0.0")  # noqa: S104 - internal container bind
    p_ws.add_argument("--port", type=int, default=9000)
    p_ws.set_defaults(func=_cmd_webhook_stub)

    p_demo = sub.add_parser("demo-generate", help="write deterministic normal + drift events")
    p_demo.add_argument("--out", default=None, help="output dir (default reports/demo)")
    p_demo.add_argument("--n", type=int, default=200)
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.set_defaults(func=_cmd_demo_generate)

    p_replay = sub.add_parser("replay", help="replay an event file through the monitor")
    p_replay.add_argument("--file", required=True, help="JSON-lines event file")
    p_replay.add_argument("--webhook-url", default=None, help="webhook stub /alert URL")
    p_replay.add_argument("--window-size", type=int, default=100)
    p_replay.add_argument("--min-samples", type=int, default=50)
    p_replay.set_defaults(func=_cmd_replay)

    p_mon = sub.add_parser("monitor", help="run the monitor (Redpanda or file source)")
    p_mon.add_argument("--brokers", default="localhost:9092")
    p_mon.add_argument("--topic", default="drifttrace.predictions")
    p_mon.add_argument("--file", default=None, help="use a file source instead of Redpanda")
    p_mon.add_argument("--webhook-url", default=None)
    p_mon.add_argument("--window-size", type=int, default=100)
    p_mon.add_argument("--min-samples", type=int, default=50)
    p_mon.add_argument("--limit", type=int, default=None)
    p_mon.set_defaults(func=_cmd_monitor)

    p_rollback = sub.add_parser("rollback", help="operator: roll serving back to a version")
    p_rollback.add_argument("--to-version", required=True)
    p_rollback.add_argument("--approve", action="store_true", help="explicit operator approval")
    p_rollback.add_argument("--approver", default=None)
    p_rollback.set_defaults(func=_cmd_rollback)

    p_explain = sub.add_parser("explain", help="SHAP (default) or LIME explanation")
    p_explain.add_argument("--income", type=float, required=True)
    p_explain.add_argument("--method", default="shap", choices=["shap", "lime"])
    p_explain.set_defaults(func=_cmd_explain)

    p_gov = sub.add_parser("governance", help="run fairness + privacy governance checks")
    p_gov.set_defaults(func=_cmd_governance)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
