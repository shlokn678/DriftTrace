"""Unit tests for root-cause-only alerting (FR-11)."""

from __future__ import annotations

import json

import pytest

from drifttrace.alerting.alerter import Alerter
from drifttrace.alerting.webhook import DeliveryResult
from drifttrace.rca.engine import RCAResult, RootCauseCandidate


class _FakeWebhook:
    def __init__(self, ok: bool = True):
        self.ok = ok
        self.sent: list[dict] = []

    def send(self, payload):  # noqa: ANN001
        self.sent.append(payload)
        return DeliveryResult(
            delivered=self.ok,
            status_code=200 if self.ok else None,
            error=None if self.ok else "down",
        )


def _rca_one_root() -> RCAResult:
    return RCAResult(
        window_id="w0",
        baseline_version="1",
        has_root_cause=True,
        root_cause_candidates=[
            RootCauseCandidate(
                node="income",
                severity=0.9,
                symptom_path=["credit_score", "risk_score"],
                evidence={},
            )
        ],
        symptoms=["credit_score", "risk_score"],
        drifted_nodes=["income", "credit_score", "risk_score"],
    )


@pytest.mark.unit
def test_exactly_one_alert_for_causal_chain() -> None:
    wh = _FakeWebhook()
    alerter = Alerter(webhook=wh, cooldown_seconds=0, now_fn=lambda: 1000.0)
    alerts = alerter.process(_rca_one_root())
    assert len(alerts) == 1
    assert alerts[0].root_cause == "income"
    # No separate alerts for the symptoms.
    assert len(wh.sent) == 1
    assert wh.sent[0]["type"] == "root_cause_alert"


@pytest.mark.unit
def test_no_alert_when_no_root_cause() -> None:
    wh = _FakeWebhook()
    alerter = Alerter(webhook=wh)
    rca = RCAResult(window_id="w0", baseline_version="1", has_root_cause=False)
    assert alerter.process(rca) == []
    assert wh.sent == []


@pytest.mark.unit
def test_cooldown_suppresses_repeat() -> None:
    wh = _FakeWebhook()
    clock = {"t": 1000.0}
    alerter = Alerter(webhook=wh, cooldown_seconds=900, now_fn=lambda: clock["t"])
    assert len(alerter.process(_rca_one_root())) == 1
    # Same root cause within cool-down -> suppressed.
    clock["t"] = 1100.0
    assert len(alerter.process(_rca_one_root())) == 0
    # After cool-down -> alert again.
    clock["t"] = 2500.0
    assert len(alerter.process(_rca_one_root())) == 1


@pytest.mark.unit
def test_alert_persisted_locally(tmp_path) -> None:
    wh = _FakeWebhook()
    path = tmp_path / "alerts.jsonl"
    alerter = Alerter(webhook=wh, cooldown_seconds=0, alert_log_path=path, now_fn=lambda: 1.0)
    alerter.process(_rca_one_root())
    assert path.exists()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["root_cause"] == "income"
    assert rec["incident_id"].startswith("inc-")


@pytest.mark.unit
def test_webhook_failure_does_not_crash() -> None:
    wh = _FakeWebhook(ok=False)
    alerter = Alerter(webhook=wh, cooldown_seconds=0, now_fn=lambda: 1.0)
    alerts = alerter.process(_rca_one_root())
    assert len(alerts) == 1
    assert alerts[0].delivered is False
    assert alerts[0].delivery_status.startswith("failed")


@pytest.mark.unit
def test_alert_payload_fields() -> None:
    wh = _FakeWebhook()
    alerter = Alerter(webhook=wh, cooldown_seconds=0, now_fn=lambda: 1.0)
    alert = alerter.process(_rca_one_root())[0]
    d = alert.to_dict()
    for key in [
        "incident_id",
        "root_cause",
        "model_version",
        "window_id",
        "symptom_path",
        "evidence",
        "timestamp",
        "delivered",
        "delivery_status",
    ]:
        assert key in d
