"""Root-cause-only alerting with cool-down and local persistence (FR-11).

Emits at most one alert per root-cause candidate per window, never for symptoms
(FR-11.1/11.3). Repeat alerts for the same still-open root cause are suppressed for a
configurable cool-down period (FR-11.4). Alerts are always persisted locally; webhook
delivery failure is recorded but never crashes monitoring (FR-11 AC-4).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from drifttrace.alerting.webhook import WebhookClient
from drifttrace.rca.engine import RCAResult

logger = logging.getLogger("drifttrace.alerting.alerter")


def _incident_id(root_cause: str, model_version: str) -> str:
    """Stable-ish incident id for a (root cause, model version) pair within a run."""
    base = f"{root_cause}:{model_version}:{uuid.uuid4().hex[:8]}"
    return "inc-" + hashlib.sha1(base.encode()).hexdigest()[:12]


@dataclass
class Alert:
    """A root-cause alert payload with delivery status."""

    incident_id: str
    root_cause: str
    model_version: str
    window_id: str
    symptom_path: list[str]
    evidence: dict
    timestamp: float
    delivered: bool = False
    delivery_status: str = "not_attempted"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CooldownState:
    """Tracks the last alert time per (root_cause, model_version) key (FR-11.4)."""

    cooldown_seconds: int
    last_alert_at: dict[str, float] = field(default_factory=dict)

    def should_suppress(self, key: str, now: float) -> bool:
        last = self.last_alert_at.get(key)
        return last is not None and (now - last) < self.cooldown_seconds

    def record(self, key: str, now: float) -> None:
        self.last_alert_at[key] = now


class Alerter:
    """Turn RCA results into root-cause-only alerts with cool-down + persistence."""

    def __init__(
        self,
        webhook: WebhookClient | None = None,
        cooldown_seconds: int = 900,
        alert_log_path: Path | None = None,
        now_fn: Callable[[], float] = time.time,  # injectable clock for deterministic tests
    ) -> None:
        self.webhook = webhook
        self.cooldown = CooldownState(cooldown_seconds=cooldown_seconds)
        self.alert_log_path = alert_log_path
        self._now = now_fn

    def process(self, rca: RCAResult) -> list[Alert]:
        """Emit alerts for the RCA root-cause candidates. Returns the alerts emitted.

        Symptoms never produce their own alert (FR-11.1/11.3). One alert per
        independent root-cause candidate, subject to cool-down.
        """
        alerts: list[Alert] = []
        if not rca.has_root_cause:
            return alerts

        now = self._now()
        for candidate in rca.root_cause_candidates:
            key = f"{candidate.node}:{rca.baseline_version}"
            if self.cooldown.should_suppress(key, now):
                logger.info("cool-down active; suppressing alert for %s", key)
                continue

            alert = Alert(
                incident_id=_incident_id(candidate.node, rca.baseline_version),
                root_cause=candidate.node,
                model_version=rca.baseline_version,
                window_id=rca.window_id,
                symptom_path=candidate.symptom_path,
                evidence=candidate.evidence,
                timestamp=now,
            )
            self._deliver(alert)
            self._persist(alert)
            self.cooldown.record(key, now)
            alerts.append(alert)

        return alerts

    def _deliver(self, alert: Alert) -> None:
        if self.webhook is None:
            alert.delivery_status = "no_webhook_configured"
            return
        payload = {"type": "root_cause_alert", **alert.to_dict()}
        result = self.webhook.send(payload)
        alert.delivered = result.delivered
        alert.delivery_status = (
            f"delivered:{result.status_code}"
            if result.delivered
            else f"failed:{result.error or result.status_code}"
        )

    def _persist(self, alert: Alert) -> None:
        if self.alert_log_path is None:
            return
        try:
            self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.alert_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict()) + "\n")
        except OSError as exc:  # persistence failure must not crash monitoring
            logger.warning("alert persistence failed: %s", exc)


# Count of suppressed alerts is exposed for metrics via the return of process() vs
# candidates; the monitor computes suppressions as (candidates - alerts_emitted).
