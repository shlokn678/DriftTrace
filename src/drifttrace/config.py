"""Shared configuration helpers for DriftTrace.

Centralizes filesystem locations and non-secret defaults so that every component
(data, features, training, drift, serving, streaming) resolves the same paths.

No secrets live here. Runtime overrides come from environment variables with safe
local defaults, per NFR-7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root.

    Resolved relative to this file: src/drifttrace/config.py -> repo root is two
    parents up from the package directory. This keeps paths stable regardless of
    the current working directory (supports NFR-4 portability).
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Canonical project paths.

    All paths derive from :attr:`root`, which defaults to the repository root and is
    overridable via ``DRIFTTRACE_ROOT``. Nothing here depends on an absolute
    developer-machine path, so the project works when copied to another directory
    (NFR-4 portability).
    """

    root: Path
    config: Path
    data: Path
    artifacts: Path
    reports: Path
    model_store: Path

    @property
    def drift_yaml(self) -> Path:
        return self.config / "drift.yaml"

    @property
    def governance_yaml(self) -> Path:
        return self.config / "governance.yaml"


def get_paths() -> Paths:
    """Build the canonical :class:`Paths`.

    ``DRIFTTRACE_ROOT`` overrides the project root; individual directories can be
    overridden independently via ``DRIFTTRACE_DATA_DIR``, ``DRIFTTRACE_ARTIFACTS_DIR``,
    ``DRIFTTRACE_REPORTS_DIR`` and ``DRIFTTRACE_MODEL_STORE``. All defaults are relative
    to the resolved root, so a copied/cloned checkout works with no configuration.
    """
    root = Path(os.environ.get("DRIFTTRACE_ROOT", str(_repo_root()))).resolve()

    def _dir(env: str, default: Path) -> Path:
        val = os.environ.get(env)
        return Path(val).resolve() if val else default

    return Paths(
        root=root,
        config=_dir("DRIFTTRACE_CONFIG_DIR", root / "config"),
        data=_dir("DRIFTTRACE_DATA_DIR", root / "data"),
        artifacts=_dir("DRIFTTRACE_ARTIFACTS_DIR", root / "artifacts"),
        reports=_dir("DRIFTTRACE_REPORTS_DIR", root / "reports"),
        model_store=_dir("DRIFTTRACE_MODEL_STORE", root / "artifacts" / "uploaded_models"),
    )


# Default random seed for reproducibility (NFR-12). Overridable per call.
DEFAULT_SEED = 42
