"""Unit tests for live-monitoring persistence (LiveStore)."""

from __future__ import annotations

import pytest

from drifttrace.streaming.live_store import LiveStore


@pytest.mark.unit
def test_status_round_trip(tmp_path) -> None:
    store = LiveStore(tmp_path / "live")
    assert store.read_status() is None
    store.write_status({"status": "RUNNING", "events_received": 42})
    got = store.read_status()
    assert got is not None
    assert got["status"] == "RUNNING"
    assert got["events_received"] == 42


@pytest.mark.unit
def test_window_history_append_and_limit(tmp_path) -> None:
    store = LiveStore(tmp_path / "live")
    assert store.read_windows() == []
    for i in range(5):
        store.append_window({"sequence": i, "status": "STABLE"})
    all_windows = store.read_windows()
    assert [w["sequence"] for w in all_windows] == [0, 1, 2, 3, 4]
    # Limit returns the most recent N in order.
    assert [w["sequence"] for w in store.read_windows(limit=2)] == [3, 4]
    assert store.latest_window()["sequence"] == 4


@pytest.mark.unit
def test_incidents_round_trip_and_default(tmp_path) -> None:
    store = LiveStore(tmp_path / "live")
    # Default shape before anything is written.
    assert store.read_incidents() == {"open": None, "history": []}
    store.write_incidents({"open": {"incident_id": "live-inc-0001"}, "history": []})
    got = store.read_incidents()
    assert got["open"]["incident_id"] == "live-inc-0001"


@pytest.mark.unit
def test_clear_removes_artifacts(tmp_path) -> None:
    store = LiveStore(tmp_path / "live")
    store.write_status({"status": "RUNNING"})
    store.append_window({"sequence": 0})
    store.write_incidents({"open": None, "history": []})
    store.clear()
    assert store.read_status() is None
    assert store.read_windows() == []
    assert store.read_incidents() == {"open": None, "history": []}


@pytest.mark.unit
def test_corrupt_status_returns_none(tmp_path) -> None:
    store = LiveStore(tmp_path / "live")
    store.status_path.write_text("{ not valid json", encoding="utf-8")
    assert store.read_status() is None
