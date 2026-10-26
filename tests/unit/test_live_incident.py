"""Unit tests for the live-monitoring incident lifecycle + window-result shaping.

These test ONLY the new orchestration logic (the incident finite-state machine and the
serialization of existing-engine output). Drift/RCA statistics are produced by the
existing engines and are exercised in the integration test.
"""

from __future__ import annotations

import pytest

from drifttrace.streaming.live import IncidentPolicy, IncidentTracker


@pytest.mark.unit
def test_incident_opens_after_two_consecutive_drifted_windows() -> None:
    tracker = IncidentTracker(IncidentPolicy(open_after=2, close_after=3))

    # One drifted window is not yet a confirmed incident (DRIFT detected, not confirmed).
    assert tracker.observe("w0", drifted=True, root_cause="mean_radius", symptom_path=[]) is None
    # Second consecutive drifted window opens the incident.
    inc = tracker.observe("w1", drifted=True, root_cause="mean_radius", symptom_path=["mean_area"])
    assert inc is not None
    assert inc.status == "OPEN"
    assert inc.root_cause == "mean_radius"
    assert inc.opened_window == "w1"


@pytest.mark.unit
def test_incident_updates_same_incident_while_drift_continues() -> None:
    tracker = IncidentTracker(IncidentPolicy(open_after=2, close_after=3))
    tracker.observe("w0", True, "r", [])
    inc1 = tracker.observe("w1", True, "r", [])
    assert inc1 is not None
    first_id = inc1.incident_id
    inc2 = tracker.observe("w2", True, "r", [])
    inc3 = tracker.observe("w3", True, "r", [])
    # Same incident object/id is updated, not a new one per window (no alert spam).
    assert inc2 is not None and inc3 is not None
    assert inc2.incident_id == first_id == inc3.incident_id
    assert inc3.last_window == "w3"
    assert inc3.drifted_windows >= 4


@pytest.mark.unit
def test_incident_closes_after_three_consecutive_stable_windows() -> None:
    tracker = IncidentTracker(IncidentPolicy(open_after=2, close_after=3))
    tracker.observe("w0", True, "r", [])
    tracker.observe("w1", True, "r", [])  # open
    assert tracker.open_incident is not None

    # Two stable windows: not yet closed.
    tracker.observe("w2", False, None, [])
    tracker.observe("w3", False, None, [])
    assert tracker.open_incident is not None
    # Third consecutive stable window closes it.
    tracker.observe("w4", False, None, [])
    assert tracker.open_incident is None
    assert len(tracker.history) == 1
    closed = tracker.history[0]
    assert closed.status == "CLOSED"
    assert closed.closed_window == "w4"


@pytest.mark.unit
def test_intermittent_drift_resets_stable_counter() -> None:
    tracker = IncidentTracker(IncidentPolicy(open_after=2, close_after=3))
    tracker.observe("w0", True, "r", [])
    tracker.observe("w1", True, "r", [])  # open
    tracker.observe("w2", False, None, [])  # 1 stable
    tracker.observe("w3", True, "r", [])  # drift resets the stable streak
    tracker.observe("w4", False, None, [])  # 1 stable again
    tracker.observe("w5", False, None, [])  # 2 stable
    # Only two consecutive stable windows since the last drift -> still open.
    assert tracker.open_incident is not None


@pytest.mark.unit
def test_single_drifted_window_does_not_open_with_default_policy() -> None:
    tracker = IncidentTracker(IncidentPolicy(open_after=2, close_after=3))
    tracker.observe("w0", True, "r", [])
    tracker.observe("w1", False, None, [])
    # A lone drifted window flanked by stable ones never opens an incident.
    assert tracker.open_incident is None
    assert tracker.history == []


@pytest.mark.unit
def test_snapshot_shape() -> None:
    tracker = IncidentTracker(IncidentPolicy(open_after=1, close_after=1))
    tracker.observe("w0", True, "root", ["sym"])
    snap = tracker.snapshot()
    assert snap["open"] is not None
    assert snap["open"]["root_cause"] == "root"
    assert isinstance(snap["history"], list)
