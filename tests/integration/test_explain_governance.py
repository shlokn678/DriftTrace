"""Integration tests for explainability (SHAP/LIME) and fairness (FR-14, FR-15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.explain.explainer import (
    lime_explain_local,
    shap_explain_local,
    shap_global_importance,
)
from drifttrace.features.transform import MODEL_FEATURES, fit_params, transform
from drifttrace.governance.config import load_governance_config
from drifttrace.governance.fairness import evaluate_fairness
from drifttrace.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def trained():
    frame = generate(GeneratorParams(n_rows=1500, seed=42))
    estimator, _ = train(frame, TrainConfig(seed=42, min_roc_auc=0.6))
    params = fit_params(frame["income"].to_numpy(dtype=float))
    return estimator, params, frame


@pytest.fixture
def background(trained):
    _, params, _ = trained
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"income": rng.lognormal(8.5, 0.5, 40)})
    return transform(frame, params)[MODEL_FEATURES]


@pytest.mark.integration
def test_shap_local_explanation(trained, background) -> None:
    estimator, params, _ = trained
    exp = shap_explain_local(estimator, 4000.0, params, model_version="1", background=background)
    assert exp.method == "shap"
    assert exp.scope == "local"
    assert exp.model_version == "1"
    assert {a.feature for a in exp.attributions} == set(MODEL_FEATURES)
    assert "not a proven causal mechanism" in exp.caveat


@pytest.mark.integration
def test_shap_global_importance(trained, background) -> None:
    estimator, _, _ = trained
    exp = shap_global_importance(estimator, background, model_version="1")
    assert exp.scope == "global"
    assert len(exp.attributions) == len(MODEL_FEATURES)


@pytest.mark.integration
def test_lime_local_explanation(trained, background) -> None:
    estimator, params, _ = trained
    exp = lime_explain_local(estimator, 4000.0, params, background, model_version="1")
    assert exp.method == "lime"
    assert exp.scope == "local"
    assert len(exp.attributions) == len(MODEL_FEATURES)


@pytest.mark.integration
def test_explain_serializable(trained, background) -> None:
    import json

    estimator, params, _ = trained
    exp = shap_explain_local(estimator, 3000.0, params, model_version="1", background=background)
    json.dumps(exp.to_dict())


@pytest.mark.integration
def test_fairness_uses_declared_attribute(trained) -> None:
    estimator, params, frame = trained
    cfg = load_governance_config()
    evidence = evaluate_fairness(estimator, frame, params, cfg)
    assert evidence.sensitive_attribute == "group"
    metrics = evidence.metrics
    assert "selection_rate" in metrics
    assert set(metrics["selection_rate"].keys()) == {"A", "B"}
    assert metrics["tpr_gap"] >= 0.0


@pytest.mark.integration
def test_fairness_rejects_missing_attribute(trained) -> None:
    estimator, params, frame = trained
    cfg = load_governance_config()
    frame_no_group = frame.drop(columns=["group"])
    with pytest.raises(ValueError, match="not present"):
        evaluate_fairness(estimator, frame_no_group, params, cfg)
