"""Streaming monitor: consume events, window them, notify (FR-13.2/13.3).

The monitor is broker-independent: it takes any :class:`EventSource` and any window
handler, so the same loop runs over file replay or Redpanda (FR-13.6). Monitoring is
asynchronous relative to serving; nothing here blocks the prediction path (FR-13.3,
NFR-2).

Phase 3 boundary: the default handler produces a NON-statistical window notification
(counts + per-feature means + sources) and posts it to the webhook stub. It does NOT
perform KS/PSI drift detection or RCA - that is Phase 4. The handler is a single,
swappable seam so Phase 4 can plug in the real detector without touching this loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from drifttrace.alerting.webhook import WebhookClient
from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.source import EventSource
from drifttrace.streaming.window import TumblingWindower, Window, WindowPolicy

logger = logging.getLogger("drifttrace.streaming.monitor")

# A window handler receives a closed window and returns a notification payload (or None
# to send nothing). Phase 4 replaces the default with KS/PSI + RCA behind this seam.
WindowHandler = Callable[[Window], dict | None]


def default_window_handler(window: Window) -> dict:
    """Phase 3 handler: summarize the window (no drift algorithm)."""
    return {
        "type": "window_summary",
        "note": "phase3-infrastructure-only; no drift detection performed",
        **window.summary(),
    }


@dataclass
class MonitorResult:
    """Aggregate outcome of a monitor run over a finite source."""

    windows_processed: int = 0
    notifications_sent: int = 0
    notifications_failed: int = 0
    window_summaries: list[dict] = field(default_factory=list)
    outcomes: list[dict] = field(default_factory=list)  # Phase 4 per-window outcomes

    def to_dict(self) -> dict:
        return {
            "windows_processed": self.windows_processed,
            "notifications_sent": self.notifications_sent,
            "notifications_failed": self.notifications_failed,
            "window_summaries": self.window_summaries,
            "outcomes": self.outcomes,
        }


class Monitor:
    """Consume events from a source, window them, and process per closed window.

    Two modes:
    - Phase 4 (preferred): pass a ``processor`` (a Phase4Processor) that runs
      drift -> RCA -> report -> root-cause alert per window.
    - Phase 3 (legacy): pass a ``handler`` + ``webhook`` to post window summaries.
    """

    def __init__(
        self,
        source: EventSource,
        webhook: WebhookClient | None = None,
        policy: WindowPolicy | None = None,
        handler: WindowHandler | None = None,
        processor: object | None = None,
    ) -> None:
        self.source = source
        self.webhook = webhook
        self.windower = TumblingWindower(policy)
        self.processor = processor
        # Only fall back to the Phase 3 summary handler when no processor is given.
        self.handler = (
            handler
            if handler is not None
            else (None if processor is not None else default_window_handler)
        )

    def _process_window(self, window: Window, result: MonitorResult) -> None:
        result.windows_processed += 1
        # Phase 4 path: the processor owns drift/RCA/report/alert.
        if self.processor is not None:
            outcome = self.processor.process(window)  # type: ignore[attr-defined]
            result.outcomes.append(outcome.to_dict())
            return
        # Phase 3 legacy path.
        if self.handler is None:
            return
        payload = self.handler(window)
        if payload is None:
            return
        result.window_summaries.append(payload)
        if self.webhook is not None:
            delivery = self.webhook.send(payload)
            if delivery.delivered:
                result.notifications_sent += 1
            else:
                result.notifications_failed += 1

    def run(self, limit: int | None = None) -> MonitorResult:
        """Run over a finite source (file replay, or a bounded Redpanda poll).

        Windows fill at the policy size; a trailing partial window is flushed so small
        demo/test streams still produce a window and a notification.
        """
        result = MonitorResult()
        event: PredictionEvent
        for event in self.source.read(limit=limit):
            closed = self.windower.add(event)
            if closed is not None:
                self._process_window(closed, result)
        tail = self.windower.flush()
        if tail is not None:
            self._process_window(tail, result)
        return result
