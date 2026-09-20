"""Unit tests for the versioned drift baseline (FR-9.2/9.3)."""

from __future__ import annotations

import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.drift.baseline import Baseline, build_baseline
from drifttrace.graph.loader import load_graph


@pytest.fixture
def graph():
    return load_graph()


@pytest.mark.unit
def test_baseline_covers_feature_nodes(graph) -> None:
    df = generate(GeneratorParams(n_rows=1000, seed=31))
    baseline = build_baseline(df, graph, model_version="1")
    # income, credit_score, risk_score are captured; prediction (model output) is not.
    assert set(baseline.nodes.keys()) == {"income", "credit_score", "risk_score"}
    assert "prediction" not in baseline.nodes


@pytest.mark.unit
def test_continuous_baseline_has_values(graph) -> None:
    df = generate(GeneratorParams(n_rows=1000, seed=31))
    baseline = build_baseline(df, graph, model_version="1")
    inc = baseline.nodes["income"]
    assert inc.kind == "continuous"
    assert len(inc.values) > 0
    assert inc.count == 1000


@pytest.mark.unit
def test_reference_sample_is_bounded(graph) -> None:
    df = generate(GeneratorParams(n_rows=8000, seed=31))
    baseline = build_baseline(df, graph, model_version="1", max_reference_samples=2000)
    assert len(baseline.nodes["income"].values) == 2000
    assert baseline.nodes["income"].count == 8000


@pytest.mark.unit
def test_baseline_roundtrip_save_load(graph, tmp_path) -> None:
    df = generate(GeneratorParams(n_rows=500, seed=31))
    baseline = build_baseline(df, graph, model_version="7")
    path = tmp_path / "baseline.json"
    baseline.save(path)
    loaded = Baseline.load(path)
    assert loaded.model_version == "7"
    assert set(loaded.nodes.keys()) == set(baseline.nodes.keys())
    assert loaded.nodes["income"].values == baseline.nodes["income"].values


@pytest.mark.unit
def test_baseline_is_versioned(graph) -> None:
    df = generate(GeneratorParams(n_rows=200, seed=31))
    b1 = build_baseline(df, graph, model_version="3")
    assert b1.model_version == "3"
