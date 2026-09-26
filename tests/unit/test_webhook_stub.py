"""Tests for the webhook stub receiver (FR-11.5) and webhook client (FR-11 AC-4)."""

from __future__ import annotations

import pytest

from drifttrace.alerting.webhook import WebhookClient
from drifttrace.alerting.webhook_stub import STORE, create_app


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    STORE.clear()
    return TestClient(create_app())


@pytest.mark.unit
def test_stub_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_stub_records_alert(client) -> None:
    resp = client.post("/alert", json={"type": "window_summary", "count": 3})
    assert resp.status_code == 200
    assert resp.json()["received"] is True
    got = client.get("/received").json()
    assert got["count"] == 1
    assert got["items"][0]["count"] == 3


@pytest.mark.unit
def test_stub_reset(client) -> None:
    client.post("/alert", json={"x": 1})
    assert client.get("/count").json()["count"] == 1
    client.post("/reset")
    assert client.get("/count").json()["count"] == 0


@pytest.mark.unit
def test_webhook_client_delivers_to_stub(monkeypatch) -> None:
    """WebhookClient posts JSON to the stub in-process via the FastAPI TestClient."""
    import httpx as httpx_mod
    from fastapi.testclient import TestClient

    STORE.clear()
    test_client = TestClient(create_app())

    # Route WebhookClient's httpx.post through the in-process TestClient (no network).
    def _fake_post(url, json=None, timeout=None):  # noqa: ANN001, A002
        return test_client.post("/alert", json=json)

    monkeypatch.setattr(httpx_mod, "post", _fake_post)
    result = WebhookClient(url="http://stub/alert").send({"type": "window_summary", "count": 5})

    assert result.delivered is True
    assert result.status_code == 200
    assert STORE.count == 1


@pytest.mark.unit
def test_webhook_client_tolerates_failure() -> None:
    """A bad URL must not raise; delivered=False is returned (FR-11 AC-4)."""
    client = WebhookClient(url="http://127.0.0.1:1/definitely-not-listening", timeout=0.2)
    result = client.send({"x": 1})
    assert result.delivered is False
    assert result.error is not None
