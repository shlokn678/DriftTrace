"""Unit tests for operator actions: rollback + retrain approval + audit (FR-12)."""

from __future__ import annotations

import pytest

from drifttrace.governance.audit import read_audit
from drifttrace.governance.operator import ApprovalRequired, retrain, rollback


@pytest.mark.unit
def test_rollback_requires_approval() -> None:
    with pytest.raises(ApprovalRequired):
        rollback("1", approved=False)


@pytest.mark.unit
def test_rollback_with_approval_audits(tmp_path) -> None:
    audit = tmp_path / "audit.jsonl"
    result = rollback("2", approved=True, approver="alice", audit_path=audit)
    assert result.ok
    assert result.detail["to_model"] == "2"
    entries = read_audit(audit)
    assert len(entries) == 1
    assert entries[0].action == "rollback"
    assert entries[0].approved is True
    assert entries[0].approver == "alice"


@pytest.mark.unit
def test_retrain_requires_approval() -> None:
    with pytest.raises(ApprovalRequired):
        retrain(approved=False)


@pytest.mark.unit
def test_retrain_with_approval_audits_only(tmp_path) -> None:
    audit = tmp_path / "audit.jsonl"
    # Retrain records the approved decision but performs no training (models are
    # uploaded, not trained by DriftTrace).
    result = retrain(approved=True, approver="bob", audit_path=audit)
    assert result.ok
    entries = read_audit(audit)
    assert len(entries) == 1
    assert entries[0].action == "retrain"
    assert entries[0].approver == "bob"


@pytest.mark.unit
def test_no_automatic_action() -> None:
    # Neither action does anything without explicit approval=True.
    for fn in (lambda: rollback("1", approved=False), lambda: retrain(approved=False)):
        with pytest.raises(ApprovalRequired):
            fn()
