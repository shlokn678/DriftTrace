"""Unit tests for the KS drift detector (FR-9.1/9.4/9.5)."""

from __future__ import annotations

import numpy as np
import pytest

from drifttrace.drift.ks import ks_test
from drifttrace.drift.verdict import Verdict


@pytest.mark.unit
def test_ks_stable_same_distribution() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 500)
    cur = rng.normal(0, 1, 500)
    res = ks_test("income", cur, base, window_id="w0", baseline_version="1", threshold=0.05)
    assert res.verdict == str(Verdict.STABLE)
    assert res.p_value is not None and res.p_value >= 0.05
    assert res.statistic is not None


@pytest.mark.unit
def test_ks_obvious_drift() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 500)
    cur = rng.normal(5, 1, 500)  # shifted far
    res = ks_test("income", cur, base, window_id="w0", baseline_version="1", threshold=0.05)
    assert res.verdict == str(Verdict.DRIFT)
    assert res.p_value is not None and res.p_value < 0.05


@pytest.mark.unit
def test_ks_insufficient_samples() -> None:
    res = ks_test(
        "income",
        [1.0, 2.0],
        list(range(100)),
        window_id="w0",
        baseline_version="1",
        min_samples=30,
    )
    assert res.verdict == str(Verdict.INSUFFICIENT_DATA)
    assert res.statistic is None and res.p_value is None
    assert res.current_n == 2


@pytest.mark.unit
def test_ks_evidence_fields() -> None:
    rng = np.random.default_rng(2)
    res = ks_test(
        "credit_score",
        rng.normal(0, 1, 100),
        rng.normal(0, 1, 100),
        window_id="w7",
        baseline_version="3",
        threshold=0.05,
    )
    d = res.to_dict()
    for key in [
        "node",
        "window_id",
        "baseline_version",
        "current_n",
        "baseline_n",
        "statistic",
        "p_value",
        "threshold",
        "verdict",
    ]:
        assert key in d
    assert d["node"] == "credit_score"
    assert d["window_id"] == "w7"
    assert d["baseline_version"] == "3"


@pytest.mark.unit
def test_ks_threshold_boundary_controls_verdict() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 400)
    cur = rng.normal(0.3, 1, 400)  # mild shift
    strict = ks_test("n", cur, base, window_id="w", baseline_version="1", threshold=0.5)
    lax = ks_test("n", cur, base, window_id="w", baseline_version="1", threshold=1e-9)
    # A higher threshold makes DRIFT more likely; a near-zero threshold makes STABLE.
    assert strict.p_value == lax.p_value  # same data, same statistic
    assert lax.verdict == str(Verdict.STABLE)
