"""End-to-end multi-root RCA verification over the REAL drift + RCA pipeline.

Uses a test-only dependency graph with two INDEPENDENT chains:

    feature_a -> feature_b
    feature_c -> feature_d

Reference and "current" datasets are generated with genuine, detectable distribution
shifts in BOTH chains, then fed through the existing KS/PSI drift engine
(``drift_report_from_frame``) and the existing RCA engine (``analyze``) - no verdicts are
hand-set, no answers are hardcoded, and the production synthetic demo bundle is untouched.

Proves:
1. The drift engine flags all four features from real data.
2. The RCA engine returns TWO independent root candidates (feature_a, feature_c).
3. feature_b / feature_d attach to the correct downstream branches.
4. The RCA result schema (to_dict) preserves BOTH roots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drifttrace.bundle.graph_json import graph_from_edges
from drifttrace.drift.baseline import build_baseline
from drifttrace.drift.config import DriftConfig, DriftThresholds
from drifttrace.drift.engine import drift_report_from_frame
from drifttrace.rca.engine import analyze


def _two_chain_graph():
    # Two disjoint chains; graph_from_edges appends the model-output node automatically.
    return graph_from_edges([["feature_a", "feature_b"], ["feature_c", "feature_d"]])


def _reference_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0.0, 1.0, n),
            "feature_b": rng.normal(5.0, 1.0, n),
            "feature_c": rng.normal(-3.0, 1.0, n),
            "feature_d": rng.normal(10.0, 2.0, n),
        }
    )


def _drifted_frame(n: int = 400) -> pd.DataFrame:
    """A genuinely shifted sample on all four features (both chains drift for real)."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(6.0, 1.0, n),  # large mean shift -> real KS/PSI drift
            "feature_b": rng.normal(11.0, 1.0, n),
            "feature_c": rng.normal(4.0, 1.0, n),
            "feature_d": rng.normal(20.0, 2.0, n),
        }
    )


@pytest.mark.integration
def test_two_independent_chains_detected_as_two_roots_end_to_end() -> None:
    graph = _two_chain_graph()
    ref = _reference_frame()
    cur = _drifted_frame()

    baseline = build_baseline(ref, graph, model_version="test-multiroot")
    config = DriftConfig(thresholds=DriftThresholds(), min_samples=30, psi_bins=10)

    # Real detector over the drifted window (no hand-set verdicts).
    drift = drift_report_from_frame(cur, baseline, config, window_id="w0")
    assert set(drift.drifted_nodes) == {"feature_a", "feature_b", "feature_c", "feature_d"}

    # Real RCA over the real drift report.
    rca = analyze(drift, graph)
    roots = {c.node for c in rca.root_cause_candidates}
    assert roots == {"feature_a", "feature_c"}, f"expected two independent roots, got {roots}"
    assert len(rca.root_cause_candidates) == 2

    by_node = {c.node: c for c in rca.root_cause_candidates}
    assert by_node["feature_a"].symptom_path == ["feature_b"]
    assert by_node["feature_c"].symptom_path == ["feature_d"]
    assert set(rca.symptoms) == {"feature_b", "feature_d"}

    # Schema preserves BOTH roots (no primary-only truncation anywhere in the pipeline).
    d = rca.to_dict()
    assert {c["node"] for c in d["root_cause_candidates"]} == {"feature_a", "feature_c"}


@pytest.mark.integration
def test_live_window_shape_preserves_multiple_roots() -> None:
    """The live window result shaping (build_window_result) preserves >1 root candidate,
    so the live API/schema and the frontend adapter can display both roots."""
    from drifttrace.streaming.event import PredictionEvent, new_event_id
    from drifttrace.streaming.live import build_window_result
    from drifttrace.streaming.processing import run_scenario_over_events

    graph = _two_chain_graph()
    ref = _reference_frame()
    cur = _drifted_frame(n=300)
    baseline = build_baseline(ref, graph, model_version="test-multiroot")
    config = DriftConfig(thresholds=DriftThresholds(), min_samples=30, psi_bins=10)

    # Build standardized events from the drifted frame and run the real Phase-4 pipeline.
    events = [
        PredictionEvent(
            event_id=new_event_id(),
            features={c: float(cur.iloc[i][c]) for c in cur.columns},
            source="test",
        )
        for i in range(len(cur))
    ]
    outcomes = run_scenario_over_events(
        events,
        baseline,
        graph,
        config,
        reports_dir=None,
        window_size=len(events),
        min_window_samples=30,
    )
    outcome = outcomes[-1]
    assert len(outcome.rca.root_cause_candidates) == 2

    window = type("W", (), {"events": events, "count": len(events)})()
    result = build_window_result(
        window,  # type: ignore[arg-type]
        outcome,
        sequence=1,
        dependencies_available=True,
        incident=None,
        graph_nodes=list(graph.feature_nodes()),
        graph_edges=graph.edge_list(),
    )
    assert set(result["root_cause_candidates"]) == {"feature_a", "feature_c"}
    assert set(result["symptoms"]) == {"feature_b", "feature_d"}
    # Edge-faithful rendering data is present and contains both chains + output edges.
    assert ["feature_a", "feature_b"] in result["graph_edges"]
    assert ["feature_c", "feature_d"] in result["graph_edges"]
