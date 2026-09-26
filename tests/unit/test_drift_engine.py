"""Unit tests for the per-node drift engine (FR-9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drifttrace.drift.baseline import Baseline, NodeBaseline
from drifttrace.drift.config import DriftConfig, DriftThresholds
from drifttrace.drift.engine import detect_drift, drift_report_from_frame
from drifttrace.drift.verdict import Verdict


def _config() -> DriftConfig:
    return DriftConfig(thresholds=DriftThresholds(), min_samples=30, psi_bins=10)


def _baseline(seed: int = 0) -> Baseline:
    rng = np.random.default_rng(seed)
    b = Baseline(model_version="1")
    b.nodes["income"] = NodeBaseline(
        node="income",
        kind="continuous",
        values=list(rng.normal(1000, 100, 1000)),
        count=1000,
    )
    return b


@pytest.mark.unit
def test_engine_stable() -> None:
    rng = np.random.default_rng(1)
    baseline = _baseline()
    values = {"income": list(rng.normal(1000, 100, 500))}
    report = detect_drift(values, {}, baseline, _config(), window_id="w0")
    assert report.nodes["income"].verdict == str(Verdict.STABLE)
    assert report.drifted_nodes == []


@pytest.mark.unit
def test_engine_drift() -> None:
    rng = np.random.default_rng(2)
    baseline = _baseline()
    values = {"income": list(rng.normal(5000, 100, 500))}  # big shift
    report = detect_drift(values, {}, baseline, _config(), window_id="w0")
    assert report.nodes["income"].verdict == str(Verdict.DRIFT)
    assert "income" in report.drifted_nodes


@pytest.mark.unit
def test_engine_insufficient() -> None:
    baseline = _baseline()
    values = {"income": [1000.0, 1001.0]}  # below min_samples
    report = detect_drift(values, {}, baseline, _config(), window_id="w0")
    assert report.nodes["income"].verdict == str(Verdict.INSUFFICIENT_DATA)


@pytest.mark.unit
def test_engine_retains_ks_and_psi_evidence() -> None:
    rng = np.random.default_rng(3)
    baseline = _baseline()
    values = {"income": list(rng.normal(1000, 100, 200))}
    report = detect_drift(values, {}, baseline, _config(), window_id="w0")
    node = report.nodes["income"]
    assert node.ks is not None
    assert node.psi is not None
    assert "statistic" in node.ks
    assert "psi" in node.psi


@pytest.mark.unit
def test_engine_from_frame() -> None:
    rng = np.random.default_rng(4)
    baseline = _baseline()
    frame = pd.DataFrame({"income": rng.normal(1000, 100, 300)})
    report = drift_report_from_frame(frame, baseline, _config(), window_id="w0")
    assert "income" in report.nodes


@pytest.mark.unit
def test_engine_serializable() -> None:
    import json

    rng = np.random.default_rng(5)
    baseline = _baseline()
    values = {"income": list(rng.normal(1000, 100, 100))}
    report = detect_drift(values, {}, baseline, _config(), window_id="w0")
    json.dumps(report.to_dict())  # must not raise
