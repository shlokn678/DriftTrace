"""Fresh-clone bootstrap for DriftTrace (local-first, model-agnostic).

Prepares the runtime environment from tracked repository files - no Docker, no broker,
no built-in model. Crucially, the bootstrap does NOT train, register, or activate any
model: a fresh install starts with NO active model. The operator uploads a model bundle
(model.pkl + reference.csv + optional graph.json) through the dashboard or API and
activates it.

Steps (all idempotent):
    1. Verify the declared config files are present.
    2. Create runtime directories (data / artifacts / reports / model store).
    3. Report readiness.

Run as a module::

    python -m drifttrace.bootstrap

Exits non-zero with a human-readable message when a prerequisite is missing.
"""

from __future__ import annotations

import argparse
import sys

from drifttrace.config import get_paths


class BootstrapError(RuntimeError):
    """A prerequisite is missing. Carries a clear, human-readable message."""


def _info(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _require_import(module: str, extra: str) -> None:
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


def bootstrap() -> int:
    """Prepare the runtime environment. Returns a process exit code (0 on success)."""
    # The serving stack needs FastAPI; model onboarding needs scikit-learn to load models.
    _require_import("fastapi", "serving")
    _require_import("sklearn", "serving")

    _ensure_dirs()

    paths = get_paths()
    _info("DriftTrace is ready. NO model is active yet.")
    _info(f"project root: {paths.root}")
    _info("start the API:  python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000")
    _info("then open the dashboard and upload a model bundle to begin monitoring.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="drifttrace-bootstrap",
        description="Prepare the DriftTrace runtime for a fresh clone (no built-in model).",
    )
    parser.parse_args(argv)
    try:
        return bootstrap()
    except BootstrapError as exc:
        print(f"\n[bootstrap] ERROR: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
