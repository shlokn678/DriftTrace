"""Webhook client for delivering notifications to the local webhook stub (FR-11.5).

A thin, resilient HTTP POST client. Delivery failure never raises to the caller (the
monitor must not crash if the webhook is down, FR-11 AC-4); failures are returned as a
result flag and logged. ``httpx`` is imported lazily so importing this module does not
require the serving extra.

Phase 3 note: this carries generic window-summary notifications end to end. Phase 4
will send root-cause alerts (with symptom path + evidence) through the same client.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("drifttrace.alerting.webhook")


@dataclass
class DeliveryResult:
    """Outcome of a webhook delivery attempt."""

    delivered: bool
    status_code: int | None = None
    error: str | None = None


class WebhookClient:
    """POST JSON payloads to a webhook URL, tolerating failures."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, payload: dict) -> DeliveryResult:
        """POST ``payload`` as JSON. Never raises; returns a :class:`DeliveryResult`."""
        try:
            import httpx  # lazy import

            resp = httpx.post(self.url, json=payload, timeout=self.timeout)
            ok = 200 <= resp.status_code < 300
            if not ok:
                logger.warning("webhook non-2xx: %s", resp.status_code)
            return DeliveryResult(delivered=ok, status_code=resp.status_code)
        except Exception as exc:  # noqa: BLE001 - resilience is the whole point
            logger.warning("webhook delivery failed: %s", exc)
            return DeliveryResult(delivered=False, error=str(exc))

    def send_json_str(self, payload: dict) -> str:
        """Helper for tests/logs: the exact JSON body that would be sent."""
        return json.dumps(payload, sort_keys=True)
