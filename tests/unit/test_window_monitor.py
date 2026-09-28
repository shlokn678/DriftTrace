"""Unit tests for windowing and the monitor (FR-13.2/13.3). No drift algorithms."""

from __future__ import annotations

import pytest

from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.monitor import Monitor, default_window_handler
from drifttrace.streaming.window import (
    TumblingWindower,
    WindowPolicy,
    windows_from_events,
)


def _events(n: int, value: float = 1000.0, source: str = "test") -> list[PredictionEvent]:
    return [
        PredictionEvent(
            event_id=f"e{i}",
            features={"feature_a": value, "feature_b": 600.0, "feature_c": 0.3},
            prediction=0,
            probability=0.3,
            source=source,
        )
        for i in range(n)
    ]


class _ListSource:
    def __init__(self, events):  # noqa: ANN001
        self._events = events

    def read(self, limit=None):  # noqa: ANN001
        for i, ev in enumerate(self._events):
            if limit is not None and i >= limit:
                return
            yield ev


@pytest.mark.unit
def test_tumbling_windower_closes_at_size() -> None:
    w = TumblingWindower(WindowPolicy(size=3, min_samples=1))
    closed = [w.add(ev) for ev in _events(6)]
    non_none = [c for c in closed if c is not None]
    assert len(non_none) == 2
    assert all(win.count == 3 for win in non_none)


@pytest.mark.unit
def test_windows_from_events_flushes_partial() -> None:
    windows = windows_from_events(_events(7), WindowPolicy(size=3, min_samples=1))
    # 3 + 3 + 1 (partial flushed)
    assert [win.count for win in windows] == [3, 3, 1]


@pytest.mark.unit
def test_window_insufficient_flag() -> None:
    windows = windows_from_events(_events(2), WindowPolicy(size=100, min_samples=50))
    assert windows[0].sufficient is False


@pytest.mark.unit
def test_window_feature_means_reflect_input() -> None:
    windows = windows_from_events(_events(4, value=2000.0), WindowPolicy(size=4, min_samples=1))
    assert windows[0].feature_means["feature_a"] == pytest.approx(2000.0)


@pytest.mark.unit
def test_monitor_processes_windows_and_calls_handler() -> None:
    source = _ListSource(_events(10))
    monitor = Monitor(source=source, webhook=None, policy=WindowPolicy(size=5, min_samples=1))
    result = monitor.run()
    assert result.windows_processed == 2
    assert len(result.window_summaries) == 2
    assert result.window_summaries[0]["type"] == "window_summary"


@pytest.mark.unit
def test_monitor_sends_to_webhook() -> None:
    class _FakeWebhook:
        def __init__(self):
            self.sent = []

        def send(self, payload):  # noqa: ANN001
            self.sent.append(payload)
            from drifttrace.alerting.webhook import DeliveryResult

            return DeliveryResult(delivered=True, status_code=200)

    webhook = _FakeWebhook()
    source = _ListSource(_events(6))
    monitor = Monitor(source=source, webhook=webhook, policy=WindowPolicy(size=3, min_samples=1))
    result = monitor.run()
    assert result.notifications_sent == 2
    assert len(webhook.sent) == 2


@pytest.mark.unit
def test_monitor_survives_webhook_failure() -> None:
    class _FailingWebhook:
        def send(self, payload):  # noqa: ANN001
            from drifttrace.alerting.webhook import DeliveryResult

            return DeliveryResult(delivered=False, error="down")

    source = _ListSource(_events(3))
    monitor = Monitor(
        source=source, webhook=_FailingWebhook(), policy=WindowPolicy(size=3, min_samples=1)
    )
    result = monitor.run()
    assert result.notifications_failed == 1
    assert result.windows_processed == 1


@pytest.mark.unit
def test_default_handler_is_non_statistical() -> None:
    windows = windows_from_events(_events(3), WindowPolicy(size=3, min_samples=1))
    payload = default_window_handler(windows[0])
    # Phase 3 guardrail: no drift scoring keys are present.
    assert payload["type"] == "window_summary"
    assert "no drift detection" in payload["note"]
    for forbidden in ("ks", "psi", "drift_score", "root_cause"):
        assert forbidden not in payload
