"""Operator actions: rollback and retrain with mandatory approval + audit (FR-12).

- No automatic rollback or retraining: both require an explicit ``approved=True``.
- Every action is recorded in the append-only audit trail (FR-15.3).

DriftTrace never retrains or rolls back a model automatically. Because models are
uploaded by the operator (DriftTrace does not train them), automated retraining is not
available; ``retrain`` records the operator's decision in the audit trail so the human
workflow and evidence are preserved, but performs no training. Rollback records the
operator's intent to switch to a different model/version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from drifttrace.governance.audit import record_action


class ApprovalRequired(RuntimeError):
    """Raised when an operator action is attempted without explicit approval."""


@dataclass
class OperatorResult:
    action: str
    ok: bool
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)


def rollback(
    to_model: str,
    approved: bool,
    approver: str | None = None,
    *,
    audit_path: Path | None = None,
) -> OperatorResult:
    """Record an operator decision to roll back to ``to_model``. Requires approval."""
    if not approved:
        raise ApprovalRequired("rollback requires explicit approval (--approve)")
    detail = {"to_model": str(to_model)}
    record_action("rollback", approved=True, approver=approver, detail=detail, path=audit_path)
    return OperatorResult(action="rollback", ok=True, detail=detail)


def retrain(
    approved: bool,
    approver: str | None = None,
    *,
    audit_path: Path | None = None,
) -> OperatorResult:
    """Record an approved retrain decision. Requires explicit approval (FR-12.2).

    DriftTrace does not train uploaded models, so no training is performed here; the
    decision is audited and the operator retrains the model outside DriftTrace, then
    uploads the new bundle.
    """
    if not approved:
        raise ApprovalRequired("retraining requires explicit approval (--approve)")
    detail: dict = {
        "approver": approver or "unspecified",
        "note": "retraining is performed outside DriftTrace; upload the new model bundle",
    }
    record_action("retrain", approved=True, approver=approver, detail=detail, path=audit_path)
    return OperatorResult(action="retrain", ok=True, detail=detail)
