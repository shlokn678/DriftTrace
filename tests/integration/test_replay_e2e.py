"""End-to-end (broker-free) Phase 3 path: demo events -> source -> monitor -> webhook.

Proves the infrastructure carries BOTH normal and simulated-drift events through the
pipeline. No drift detection is performed (Phase 3 boundary).
"""

from __future__ import annotations

import pytest

from drifttrace.alerting.webhook_stub import STORE, create_app
from drifttrace.streaming.demo import DRIFT, NORMAL, generate_events, write_events
from drifttrace.streaming.monitor import Monitor
from drifttrace.streaming.source import FileReplaySource
from drifttrace.streaming.window import WindowPolicy


class _StubWebhook:
    """WebhookClient-compatible object that posts into the stub app in-process."""

    def __init__(self, app) -> None:  # noqa: ANN001
        from fastapi.testclient import TestClient

        self._client = TestClient(app)

    def send(self, payload):  # noqa: ANN001
        from drifttrace.alerting.webhook import DeliveryResult

        resp = self._client.post("/alert", json=payload)
        return DeliveryResult(delivered=resp.status_code == 200, status_code=resp.status_code)


@pytest.mark.integration
@pytest.mark.parametrize("scenario", [NORMAL, DRIFT])
def test_replay_scenario_reaches_webhook(scenario, tmp_path) -> None:
    STORE.clear()
    app = create_app()

    # 1. Deterministic demo events -> file (the "event source").
    events = generate_events(scenario, n=120, seed=7, model_version="1")
    path = write_events(events, tmp_path / f"events_{scenario}.jsonl")

    # 2. Source -> monitor/window -> webhook stub.
    monitor = Monitor(
        source=FileReplaySource(path),
        webhook=_StubWebhook(app),
        policy=WindowPolicy(size=50, min_samples=10),
    )
    result = monitor.run()

    # 3. Windows were processed and the webhook stub received them.
    assert result.windows_processed >= 2
    assert result.notifications_sent == result.windows_processed
    assert STORE.count == result.windows_processed
    # Proof of receipt carries the scenario source tag.
    tags = {tag for item in STORE.items for tag in item["sources"]}
    assert f"replay:{scenario}" in tags


@pytest.mark.integration
def test_normal_and_drift_windows_differ(tmp_path) -> None:
    """The infrastructure carries a visible difference (no drift algorithm involved)."""
    normal_windows = _window_means(NORMAL, tmp_path)
    drift_windows = _window_means(DRIFT, tmp_path)
    assert drift_windows[0]["income"] > normal_windows[0]["income"] * 5


def _window_means(scenario, tmp_path):  # noqa: ANN001
    from drifttrace.streaming.window import windows_from_events

    events = generate_events(scenario, n=100, seed=7)
    windows = windows_from_events(events, WindowPolicy(size=50, min_samples=10))
    return [w.feature_means for w in windows]
