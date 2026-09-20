"""Lifecycle-evidence helpers (FR-17, FR-5.2).

Collects the identifiers that make a run reproducible: the code commit, a dataset
version, and a run identifier. These are attached to MLflow runs and monitoring
reports so any run can be traced to its five lifecycle-evidence artifacts.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from dataclasses import asdict, dataclass


def git_commit() -> str:
    """Return the current git commit SHA, or 'unknown' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def dataset_version(data_bytes: bytes) -> str:
    """Return a content-hash dataset version (stable for identical data)."""
    return "sha256:" + hashlib.sha256(data_bytes).hexdigest()


def new_run_id() -> str:
    """Generate a unique pipeline-execution identifier."""
    return uuid.uuid4().hex


@dataclass
class LifecycleEvidence:
    """The five lifecycle-evidence anchors for a run (FR-17.1)."""

    dataset_version: str
    code_commit: str
    pipeline_execution_id: str
    model_version: str | None = None
    monitoring_report_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
