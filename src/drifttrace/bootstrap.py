"""Fresh-clone bootstrap for DriftTrace (Phase 5).

Reconstructs everything the normal local/demo workflow needs, deterministically, from
tracked repository files - no Docker, no broker, no copied runtime artifacts, no
machine-specific paths. Safe to re-run: existing state is reused unless ``--force``.

Steps (each idempotent):
    1. Create runtime directories (data / artifacts / reports / model store).
    2. Generate the synthetic dataset if missing.
    3. Validate the dataset against the declared schema.
    4. Train + register the loan model and write the drift baseline + transform params,
       unless a registered model + baseline already exist.
    5. Verify the resulting state and report a concise summary.

Run as a module::

    python -m drifttrace.bootstrap            # normal (reuse existing state)
    python -m drifttrace.bootstrap --force    # rebuild dataset + model from scratch

Exits non-zero with a human-readable message when a prerequisite is missing.
"""

from __future__ import annotations

import argparse
import sys

from drifttrace.config import DEFAULT_SEED, get_paths


class BootstrapError(RuntimeError):
    """A prerequisite is missing or a bootstrap step failed. Carries a clear message."""


def _info(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _require_import(module: str, extra: str) -> None:
    """Fail with an actionable message if a required dependency is not installed."""
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise BootstrapError(
            f"Missing dependency '{module}'. Install the project with the required extras:\n"
            f'    python -m pip install -e ".[{extra}]"'
        )


def _ensure_dirs() -> None:
    paths = get_paths()
    for label, path in (
        ("data", paths.data),
        ("artifacts", paths.artifacts),
        ("reports", paths.reports),
        ("model store", paths.model_store),
    ):
        path.mkdir(parents=True, exist_ok=True)
        _info(f"runtime dir ready: {label} -> {path}")


def _ensure_config_present() -> None:
    """The declared config files are tracked in the repo; verify they exist."""
    paths = get_paths()
    required = {
        "schema.yaml": paths.schema_yaml,
        "graph.yaml": paths.graph_yaml,
        "drift.yaml": paths.drift_yaml,
        "governance.yaml": paths.governance_yaml,
    }
    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        raise BootstrapError(
            "Declared config files are missing from config/: "
            + ", ".join(sorted(missing))
            + ". These are tracked in the repository - ensure you cloned the full repo."
        )


def _ensure_dataset(*, n_rows: int, seed: int, force: bool) -> None:
    paths = get_paths()
    dataset = paths.data / "dataset.csv"
    if dataset.exists() and not force:
        _info(f"dataset present: {dataset}")
        return
    from drifttrace.data.build import build_dataset

    _info(f"generating synthetic dataset ({n_rows} rows, seed {seed}) ...")
    build_dataset(n_rows=n_rows, seed=seed)
    if not dataset.exists():
        raise BootstrapError(f"dataset generation did not produce {dataset}")
    _info(f"dataset written: {dataset}")


def _validate_dataset() -> None:
    from drifttrace.orchestration.stages import ValidationFailed, validate_stage

    _info("validating dataset against the declared schema ...")
    try:
        validate_stage()
    except ValidationFailed as exc:
        raise BootstrapError(f"dataset failed schema validation: {exc}") from exc
    _info("dataset passed schema validation")


def _model_registered() -> bool:
    from drifttrace.training.registry import latest_model_version

    try:
        return latest_model_version() is not None
    except Exception:  # noqa: BLE001 - a missing/empty MLflow store means "not registered"
        return False


def _ensure_model(*, seed: int, min_roc_auc: float, force: bool) -> None:
    paths = get_paths()
    baseline = paths.artifacts / "baseline.json"
    tparams = paths.artifacts / "transform_params.json"

    if not force and _model_registered() and baseline.exists() and tparams.exists():
        _info("registered model + baseline + transform params present; skipping training")
        return

    from drifttrace.training.pipeline import run_training_pipeline
    from drifttrace.training.train import TrainConfig

    _info("training + registering the loan model (deterministic) ...")
    result = run_training_pipeline(config=TrainConfig(seed=seed, min_roc_auc=min_roc_auc))
    if result.model_version is None:
        raise BootstrapError(
            "training completed but the model was not registered "
            f"(ROC-AUC {result.roc_auc:.3f} vs gate {min_roc_auc}). "
            "Lower --min-roc-auc or check the dataset."
        )
    for label, path in (("baseline", baseline), ("transform params", tparams)):
        if not path.exists():
            raise BootstrapError(f"training did not write the {label} artifact: {path}")
    _info(f"model registered: version {result.model_version} (ROC-AUC {result.roc_auc:.3f})")


def _verify() -> None:
    paths = get_paths()
    if not (paths.data / "dataset.csv").exists():
        raise BootstrapError("verification failed: dataset.csv missing")
    if not (paths.artifacts / "baseline.json").exists():
        raise BootstrapError("verification failed: baseline.json missing")
    if not _model_registered():
        raise BootstrapError("verification failed: no registered model version")
    _info("verification passed: dataset, baseline, and registered model are present")


def bootstrap(
    *,
    n_rows: int = 5000,
    seed: int = DEFAULT_SEED,
    min_roc_auc: float = 0.6,
    force: bool = False,
) -> int:
    """Run the full bootstrap. Returns a process exit code (0 on success)."""
    _require_import("sklearn", "serving")
    _require_import("mlflow", "tracking")

    _ensure_config_present()
    _ensure_dirs()
    _ensure_dataset(n_rows=n_rows, seed=seed, force=force)
    _validate_dataset()
    _ensure_model(seed=seed, min_roc_auc=min_roc_auc, force=force)
    _verify()

    paths = get_paths()
    _info("DriftTrace is ready.")
    _info(f"project root: {paths.root}")
    _info("start the API:  python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="drifttrace-bootstrap",
        description="Reconstruct DriftTrace runtime state for a fresh clone (local-first).",
    )
    parser.add_argument("--n-rows", type=int, default=5000, help="synthetic dataset rows")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic seed")
    parser.add_argument(
        "--min-roc-auc", type=float, default=0.6, help="minimum ROC-AUC gate for registration"
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild dataset + model even if present"
    )
    args = parser.parse_args(argv)

    try:
        return bootstrap(
            n_rows=args.n_rows,
            seed=args.seed,
            min_roc_auc=args.min_roc_auc,
            force=args.force,
        )
    except BootstrapError as exc:
        print(f"\n[bootstrap] ERROR: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
