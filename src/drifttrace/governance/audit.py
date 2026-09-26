"""Append-only audit trail for operator actions (FR-12, FR-15.3).

Records who/what approved rollback and retraining actions. Backed by a JSON-lines file
so entries are immutable-by-convention and easily reviewed. Part of the MLflow/DVC
audit story (FR-15.3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from drifttrace.config import get_paths


@dataclass
class AuditEntry:
    action: str  # "rollback" | "retrain" | ...
    approved: bool
    approver: str | None
    detail: dict
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _audit_path(path: Path | None) -> Path:
    return path or (get_paths().reports / "audit_log.jsonl")


def record_action(
    action: str,
    approved: bool,
    approver: str | None,
    detail: dict,
    *,
    path: Path | None = None,
) -> AuditEntry:
    """Append an audit entry and return it."""
    entry = AuditEntry(
        action=action,
        approved=approved,
        approver=approver,
        detail=detail,
        timestamp=datetime.now(UTC).isoformat(),
    )
    out = _audit_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict()) + "\n")
    return entry


def read_audit(path: Path | None = None) -> list[AuditEntry]:
    """Read all audit entries (for tests / review)."""
    out = _audit_path(path)
    if not out.exists():
        return []
    entries: list[AuditEntry] = []
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        entries.append(AuditEntry(**d))
    return entries
