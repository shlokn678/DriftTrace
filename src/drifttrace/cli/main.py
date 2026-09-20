"""DriftTrace command-line interface (FR-6.4, FR-12.4).

Thin wrapper over library functions so the whole pipeline runs without Airflow or a
broker. Phase 1 provides ``build-dataset`` and ``run-pipeline``; later phases add
``serve``, ``monitor``, ``rca show``, ``rollback``, ``retrain``, and ``inject``.
"""

from __future__ import annotations

import argparse
import json
import sys

from drifttrace.data.build import build_dataset
from drifttrace.training.pipeline import run_training_pipeline
from drifttrace.training.train import TrainConfig


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    manifest = build_dataset(n_rows=args.n_rows, seed=args.seed)
    print(json.dumps({"stage": "build-dataset", **_manifest_summary(manifest)}, indent=2))
    return 0


def _manifest_summary(manifest: dict) -> dict:
    return {
        "dataset_path": manifest["dataset_path"],
        "dataset_version": manifest["dataset_version"],
        "n_rows": manifest["n_rows"],
        "validation_passed": manifest["validation"]["passed"],
    }


def _cmd_run_pipeline(args: argparse.Namespace) -> int:
    config = TrainConfig(model=args.model, seed=args.seed, min_roc_auc=args.min_roc_auc)
    result = run_training_pipeline(config=config)
    print(json.dumps({"stage": "run-pipeline", **result.to_dict()}, indent=2))
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
