"""DriftTrace command-line interface (model-agnostic, local-first).

A thin wrapper over library functions. Commands:

- ``serve``         run the FastAPI service (upload + activate a model, then predict/monitor)
- ``webhook-stub``  run the local webhook stub receiver
- ``replay``        replay a JSON-lines prediction-event file through the monitor
- ``monitor``       run the monitor against a file source (or an optional Redpanda broker)
- ``rollback``      operator: record an approved rollback decision (audited)
- ``retrain``       operator: record an approved retrain decision (audited; no training)

There is no built-in model, dataset, or training pipeline: models are uploaded as
bundles through the API/dashboard. Monitoring commands operate on the active model's
baseline + optional graph, which live in the running service; ``replay``/``monitor``
here are for advanced/offline use against an explicit baseline is out of scope for the
MVP CLI and are provided as event transport utilities.
"""

from __future__ import annotations

import argparse
import json
import sys

from drifttrace.governance.operator import ApprovalRequired


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


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


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    from drifttrace.bootstrap import main as bootstrap_main

    return bootstrap_main([])


def _cmd_rollback(args: argparse.Namespace) -> int:
    from drifttrace.governance.operator import rollback

    try:
        result = rollback(args.to_model, approved=args.approve, approver=args.approver)
    except ApprovalRequired as exc:
        _emit({"stage": "rollback", "ok": False, "detail": {"error": str(exc)}})
        return 2
    _emit({"stage": "rollback", **result.to_dict()})
    return 0


def _cmd_retrain(args: argparse.Namespace) -> int:
    from drifttrace.governance.operator import retrain

    try:
        result = retrain(approved=args.approve, approver=args.approver)
    except ApprovalRequired as exc:
        _emit({"stage": "retrain", "ok": False, "detail": {"error": str(exc)}})
        return 2
    _emit({"stage": "retrain", **result.to_dict()})
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    _emit(
        {
            "stage": "replay",
            "ok": False,
            "detail": {
                "error": (
                    "Offline replay requires an active model's baseline + graph, which live "
                    "in the running service. Use the dashboard 'Run Drift Test' or POST "
                    "/demo/run-drift-test against the active model instead."
                )
            },
        }
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drifttrace", description="DriftTrace CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser("bootstrap", help="prepare runtime dirs (no model is created)")
    p_boot.set_defaults(func=_cmd_bootstrap)

    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_ws = sub.add_parser("webhook-stub", help="run the local webhook stub receiver")
    p_ws.add_argument("--host", default="127.0.0.1")
    p_ws.add_argument("--port", type=int, default=9000)
    p_ws.set_defaults(func=_cmd_webhook_stub)

    p_replay = sub.add_parser("replay", help="(advanced) replay an event file")
    p_replay.add_argument("--file", required=True)
    p_replay.set_defaults(func=_cmd_replay)

    p_rollback = sub.add_parser("rollback", help="operator: record an approved rollback")
    p_rollback.add_argument("--to-model", required=True)
    p_rollback.add_argument("--approve", action="store_true", help="explicit operator approval")
    p_rollback.add_argument("--approver", default=None)
    p_rollback.set_defaults(func=_cmd_rollback)

    p_retrain = sub.add_parser("retrain", help="operator: record an approved retrain decision")
    p_retrain.add_argument("--approve", action="store_true", help="explicit operator approval")
    p_retrain.add_argument("--approver", default=None)
    p_retrain.set_defaults(func=_cmd_retrain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
