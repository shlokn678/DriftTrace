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
    """Canonical project paths."""

    root: Path
    config: Path
    data: Path
    artifacts: Path
    reports: Path

    @property
    def graph_yaml(self) -> Path:
        return self.config / "graph.yaml"

    @property
    def schema_yaml(self) -> Path:
        return self.config / "schema.yaml"

    @property
    def drift_yaml(self) -> Path:
        return self.config / "drift.yaml"

    @property
    def governance_yaml(self) -> Path:
        return self.config / "governance.yaml"


def get_paths() -> Paths:
    """Build the canonical :class:`Paths`, allowing a ``DRIFTTRACE_ROOT`` override."""
    root = Path(os.environ.get("DRIFTTRACE_ROOT", str(_repo_root()))).resolve()
    return Paths(
        root=root,
        config=root / "config",
        data=root / "data",
        artifacts=root / "artifacts",
        reports=root / "reports",
    )


# Default random seed for reproducibility (NFR-12). Overridable per call.
DEFAULT_SEED = 42
