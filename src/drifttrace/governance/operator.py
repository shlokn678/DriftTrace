"""Operator actions: rollback and retrain with mandatory approval + audit (FR-12).

- No automatic rollback or retraining: both require an explicit ``approved=True``.
- Rollback points serving at a previously registered model version (FR-12.3).
- Every action is recorded in the append-only audit trail (FR-15.3).

Rollback here records the operator decision and the target version in the audit trail
and returns the target version to switch to; the serving layer reads the pinned version
via ``DRIFTTRACE_MODEL_VERSION``. This keeps the action deterministic and testable
without requiring a live registry mutation.
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
    to_version: str,
    approved: bool,
    approver: str | None = None,
    *,
    audit_path: Path | None = None,
) -> OperatorResult:
    """Roll serving back to ``to_version``. Requires explicit approval (FR-12.2)."""
    if not approved:
        raise ApprovalRequired("rollback requires explicit approval (--approve)")
    detail = {"to_version": str(to_version), "serving_env": "DRIFTTRACE_MODEL_VERSION"}
    record_action("rollback", approved=True, approver=approver, detail=detail, path=audit_path)
    return OperatorResult(action="rollback", ok=True, detail=detail)


def retrain(
    approved: bool,
    approver: str | None = None,
    *,
    dataset_path: Path | None = None,
    tracking_uri: str | None = None,
    audit_path: Path | None = None,
    run_pipeline: bool = True,
) -> OperatorResult:
    """Retrain + register a new model version. Requires explicit approval (FR-12.2)."""
    if not approved:
        raise ApprovalRequired("retraining requires explicit approval (--approve)")

    detail: dict = {"approver": approver or "unspecified"}
    if run_pipeline:
        from drifttrace.training.pipeline import run_training_pipeline

        result = run_training_pipeline(dataset_path=dataset_path, tracking_uri=tracking_uri)
        detail.update(
            {
                "model_version": result.model_version,
                "passed_gate": result.passed_gate,
                "roc_auc": result.roc_auc,
            }
        )
    record_action("retrain", approved=True, approver=approver, detail=detail, path=audit_path)
    return OperatorResult(action="retrain", ok=True, detail=detail)
