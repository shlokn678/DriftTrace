"""End-to-end drift-injection scenarios through the REAL KS/PSI + RCA (FR-18, FR-9, FR-10).

The injected scenarios only change feature values; the actual detectors determine drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drifttrace.alerting.alerter import Alerter
from drifttrace.drift.baseline import build_baseline
from drifttrace.drift.config import DriftConfig, DriftThresholds
from drifttrace.features.transform import compute_credit_score, compute_risk_score, fit_params
from drifttrace.graph.loader import load_graph
from drifttrace.streaming.demo import (
    CONTROL,
    INCOME_ANNUAL,
    MID_CHAIN,
    TWO_ROOTS,
    generate_events,
)
from drifttrace.streaming.processing import run_scenario_over_events


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture(scope="module")
def baseline():
    """Baseline built from a large training-like sample (matches the control income)."""
    rng = np.random.default_rng(7)
    income = rng.lognormal(mean=8.5, sigma=0.5, size=4000)
    params = fit_params(income)
    credit = compute_credit_score(income, params)
    risk = compute_risk_score(credit, params)
    frame = pd.DataFrame({"income": income, "credit_score": credit, "risk_score": risk})
    return build_baseline(frame, load_graph(), model_version="1")


def _config() -> DriftConfig:
    return DriftConfig(thresholds=DriftThresholds(), min_samples=30, psi_bins=10)


def _run(scenario, baseline, graph, tmp_path):
    events = generate_events(scenario, n=300, seed=7, model_version="1")
    alerter = Alerter(webhook=None, cooldown_seconds=0, now_fn=lambda: 1.0)
    outcomes = run_scenario_over_events(
        events,
        baseline,
        graph,
        _config(),
        alerter=alerter,
        reports_dir=tmp_path,
        window_size=300,
        min_window_samples=30,
    )
    assert len(outcomes) == 1
    return outcomes[0]


@pytest.mark.integration
def test_scenario_control_no_drift(baseline, graph, tmp_path) -> None:
    out = _run(CONTROL, baseline, graph, tmp_path)
    assert out.drift.drifted_nodes == []
    assert out.rca.has_root_cause is False
    assert out.alerts == []


@pytest.mark.integration
def test_scenario_income_annual_root_cause(baseline, graph, tmp_path) -> None:
    out = _run(INCOME_ANNUAL, baseline, graph, tmp_path)
    # income must be detected as drifted and be THE root cause.
    assert "income" in out.drift.drifted_nodes
    roots = [c.node for c in out.rca.root_cause_candidates]
    assert roots == ["income"]
    # Downstream nodes that also drift are symptoms, not roots.
    for sym in out.rca.symptoms:
        assert sym in ("credit_score", "risk_score")
    # Exactly one root-cause alert for the chain.
    assert len(out.alerts) == 1
    assert out.alerts[0]["root_cause"] == "income"


@pytest.mark.integration
def test_scenario_mid_chain(baseline, graph, tmp_path) -> None:
    out = _run(MID_CHAIN, baseline, graph, tmp_path)
    # income should NOT be the root (it is unshifted); credit_score is the earliest root.
    roots = [c.node for c in out.rca.root_cause_candidates]
    assert "credit_score" in roots
    assert "income" not in out.drift.drifted_nodes
    assert len(out.alerts) == 1
    assert out.alerts[0]["root_cause"] == "credit_score"


@pytest.mark.integration
def test_scenario_two_roots(baseline, graph, tmp_path) -> None:
    """Two independent roots require independent source branches.

    The production chain income->credit_score->risk_score is LINEAR, so every drifted
    downstream node has income as a transitive drifted ancestor and is correctly a
    SYMPTOM, never a co-equal root (verified here). The RCA engine's co-equal-root
    logic on a genuinely branched graph is covered by tests/unit/test_rca.py
    (test_two_independent_roots). This asserts the honest linear-chain behaviour.
    """
    out = _run(TWO_ROOTS, baseline, graph, tmp_path)
    # income is shifted (drifts) and risk_score is independently perturbed (drifts),
    # but credit_score stays near baseline. Because income is a transitive ancestor of
    # risk_score, the single earliest supported root is income; risk_score is a symptom.
    roots = [c.node for c in out.rca.root_cause_candidates]
    assert roots == ["income"]
    assert "risk_score" in out.rca.symptoms
    assert "risk_score" in out.drift.drifted_nodes  # detector really flagged it
    assert len(out.alerts) == 1
    assert out.alerts[0]["root_cause"] == "income"


@pytest.mark.integration
def test_scenarios_deterministic(baseline, graph, tmp_path) -> None:
    a = generate_events(INCOME_ANNUAL, n=100, seed=7)
    b = generate_events(INCOME_ANNUAL, n=100, seed=7)
    assert [e.to_json() for e in a] == [e.to_json() for e in b]
