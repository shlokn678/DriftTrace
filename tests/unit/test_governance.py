"""Unit tests for governance: privacy deny-list + config (FR-15)."""

from __future__ import annotations

import pytest

from drifttrace.governance.audit import read_audit, record_action
from drifttrace.governance.config import load_governance_config
from drifttrace.governance.privacy import check_payload, check_records


@pytest.mark.unit
def test_governance_config_declares_sensitive_attribute() -> None:
    cfg = load_governance_config()
    assert cfg.sensitive_attribute == "group"  # declared, not inferred
    assert "email" in cfg.pii_deny_list


@pytest.mark.unit
def test_privacy_passes_clean_payload() -> None:
    cfg = load_governance_config()
    payload = {"income": 1000.0, "credit_score": 600.0, "prediction": 1}
    result = check_payload(payload, cfg)
    assert result.passed
    assert result.violations == []


@pytest.mark.unit
def test_privacy_flags_pii_field() -> None:
    cfg = load_governance_config()
    payload = {"income": 1000.0, "email": "x@example.com", "ssn": "000"}
    result = check_payload(payload, cfg)
    assert not result.passed
    assert "email" in result.violations
    assert "ssn" in result.violations


@pytest.mark.unit
def test_privacy_checks_nested_and_lists() -> None:
    cfg = load_governance_config()
    payload = {"batch": [{"features": {"phone": "555"}}]}
    result = check_payload(payload, cfg)
    assert not result.passed
    assert "phone" in result.violations


@pytest.mark.unit
def test_privacy_check_records() -> None:
    cfg = load_governance_config()
    records = [
        {"income": 1.0, "prediction": 0},
        {"income": 2.0, "customer_id": "abc"},  # PII
    ]
    result = check_records(records, cfg)
    assert not result.passed
    assert "customer_id" in result.violations


@pytest.mark.unit
def test_audit_trail_roundtrip(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    record_action("retrain", approved=True, approver="alice", detail={"v": 2}, path=path)
    record_action("rollback", approved=True, approver="bob", detail={"to": 1}, path=path)
    entries = read_audit(path)
    assert len(entries) == 2
    assert entries[0].action == "retrain"
    assert entries[0].approver == "alice"
    assert entries[1].action == "rollback"
