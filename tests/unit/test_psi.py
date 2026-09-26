"""Unit tests for the PSI drift detector (FR-9.1/9.4/9.5)."""

from __future__ import annotations

import numpy as np
import pytest

from drifttrace.drift.psi import psi_categorical, psi_continuous
from drifttrace.drift.verdict import Verdict


@pytest.mark.unit
def test_psi_stable() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 2000)
    cur = rng.normal(0, 1, 2000)
    res = psi_continuous("income", cur, base, window_id="w", baseline_version="1")
    assert res.psi is not None and res.psi < 0.1
    assert res.verdict == str(Verdict.STABLE)


@pytest.mark.unit
def test_psi_drift() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)  # large shift -> PSI large
    res = psi_continuous("income", cur, base, window_id="w", baseline_version="1")
    assert res.psi is not None and res.psi >= 0.2
    assert res.verdict == str(Verdict.DRIFT)


@pytest.mark.unit
def test_psi_warning_band() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 5000)
    # Search for a shift that lands PSI in [0.1, 0.2). Small shift ~0.3-0.5 sigma.
    found_warning = False
    for shift in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        cur = rng.normal(shift, 1, 5000)
        res = psi_continuous("n", cur, base, window_id="w", baseline_version="1")
        if res.verdict == str(Verdict.WARNING):
            found_warning = True
            assert 0.1 <= res.psi < 0.2
            break
    assert found_warning, "expected some small shift to land in the WARNING band"


@pytest.mark.unit
def test_psi_insufficient_samples() -> None:
    res = psi_continuous(
        "n", [1.0, 2.0], list(range(100)), window_id="w", baseline_version="1", min_samples=30
    )
    assert res.psi is None
    assert res.verdict == str(Verdict.INSUFFICIENT_DATA)


@pytest.mark.unit
def test_psi_zero_proportion_edge_case_is_finite() -> None:
    # Current window entirely outside baseline range -> some bins have zero baseline or
    # current proportion; epsilon smoothing must keep PSI finite.
    base = np.linspace(0, 1, 500)
    cur = np.linspace(100, 101, 500)
    res = psi_continuous("n", cur, base, window_id="w", baseline_version="1")
    assert res.psi is not None
    assert np.isfinite(res.psi)
    assert res.verdict == str(Verdict.DRIFT)


@pytest.mark.unit
def test_psi_categorical_stable_and_drift() -> None:
    base_props = {"A": 0.65, "B": 0.35}
    stable_cur = ["A"] * 65 + ["B"] * 35
    res_stable = psi_categorical(
        "group",
        stable_cur,
        base_props,
        window_id="w",
        baseline_version="1",
        baseline_n=1000,
        min_samples=10,
    )
    assert res_stable.verdict == str(Verdict.STABLE)

    drift_cur = ["B"] * 90 + ["A"] * 10
    res_drift = psi_categorical(
        "group",
        drift_cur,
        base_props,
        window_id="w",
        baseline_version="1",
        baseline_n=1000,
        min_samples=10,
    )
    assert res_drift.verdict == str(Verdict.DRIFT)


@pytest.mark.unit
def test_psi_evidence_fields() -> None:
    rng = np.random.default_rng(4)
    res = psi_continuous(
        "risk_score", rng.random(200), rng.random(200), window_id="w1", baseline_version="2"
    )
    d = res.to_dict()
    for key in [
        "node",
        "psi",
        "bins",
        "psi_warning",
        "psi_drift",
        "verdict",
        "current_n",
        "baseline_n",
        "window_id",
        "baseline_version",
    ]:
        assert key in d
    assert len(d["bins"]) > 0
