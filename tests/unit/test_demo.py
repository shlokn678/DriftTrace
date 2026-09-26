"""Tests for the deterministic demo/replay data (Phase 3 infrastructure only)."""

from __future__ import annotations

import pytest

from drifttrace.streaming.demo import (
    DRIFT,
    NORMAL,
    generate_events,
    write_demo_datasets,
)
from drifttrace.streaming.window import WindowPolicy, windows_from_events


@pytest.mark.unit
def test_generate_normal_is_deterministic() -> None:
    a = generate_events(NORMAL, n=50, seed=7)
    b = generate_events(NORMAL, n=50, seed=7)
    assert [e.to_json() for e in a] == [e.to_json() for e in b]


@pytest.mark.unit
def test_drift_events_differ_from_normal() -> None:
    normal = generate_events(NORMAL, n=100, seed=7)
    drift = generate_events(DRIFT, n=100, seed=7)
    mean_income_normal = sum(e.features["income"] for e in normal) / len(normal)
    mean_income_drift = sum(e.features["income"] for e in drift) / len(drift)
    # Simulated drift annualizes income; the window means must differ noticeably.
    assert mean_income_drift > mean_income_normal * 5


@pytest.mark.unit
def test_unknown_scenario_raises() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        generate_events("weird", n=1)


@pytest.mark.unit
def test_write_demo_datasets(tmp_path) -> None:
    # Phase 4 harness writes all four scenario files.
    from drifttrace.streaming.demo import CONTROL, INCOME_ANNUAL, MID_CHAIN, TWO_ROOTS

    paths = write_demo_datasets(tmp_path, n=30, seed=7)
    for scenario in (CONTROL, INCOME_ANNUAL, MID_CHAIN, TWO_ROOTS):
        assert scenario in paths
        assert paths[scenario].exists()


@pytest.mark.unit
def test_demo_events_flow_through_windowing() -> None:
    """Deterministic path: sample events -> windows (proves infra carries events)."""
    for scenario in (NORMAL, DRIFT):
        events = generate_events(scenario, n=100, seed=7)
        windows = windows_from_events(events, WindowPolicy(size=50, min_samples=10))
        assert len(windows) == 2
        assert all(w.sufficient for w in windows)
        assert "income" in windows[0].feature_means
