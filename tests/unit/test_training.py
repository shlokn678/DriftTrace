"""Unit tests for training and evaluation (FR-4, FR-15.4)."""

from __future__ import annotations

import numpy as np
import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.features.transform import MODEL_FEATURES
from drifttrace.training.evaluate import compute_fairness, compute_metrics
from drifttrace.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def dataset():
    return generate(GeneratorParams(n_rows=4000, seed=21))


@pytest.mark.unit
def test_train_returns_metrics_and_passes_gate(dataset) -> None:
    _, result = train(dataset, TrainConfig(model="gradient_boosting", seed=1))
    assert result.metrics.roc_auc > 0.5
    assert result.passed_gate is (result.metrics.roc_auc >= result.config.min_roc_auc)
    for name in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]:
        assert name in result.metrics.to_dict()


@pytest.mark.unit
def test_training_is_deterministic(dataset) -> None:
    _, r1 = train(dataset, TrainConfig(seed=3))
    _, r2 = train(dataset, TrainConfig(seed=3))
    assert r1.metrics.to_dict() == r2.metrics.to_dict()


@pytest.mark.unit
def test_gate_fails_when_threshold_impossible(dataset) -> None:
    _, result = train(dataset, TrainConfig(min_roc_auc=1.01))
    assert result.passed_gate is False


@pytest.mark.unit
def test_pipeline_predicts_from_raw_income(dataset) -> None:
    """The fitted pipeline consumes model features derived from raw income."""
    estimator, _ = train(dataset, TrainConfig(seed=5))
    from drifttrace.features.transform import fit_params, transform

    tparams = fit_params(dataset["income"].to_numpy(dtype=float))
    featured = transform(dataset[["income"]].head(3), tparams)
    prob = estimator.predict_proba(featured[MODEL_FEATURES])[:, 1]
    assert prob.shape == (3,)
    assert np.all((prob >= 0) & (prob <= 1))


@pytest.mark.unit
def test_logistic_regression_model_option(dataset) -> None:
    _, result = train(dataset, TrainConfig(model="logistic_regression", seed=7))
    assert result.config.model == "logistic_regression"
    assert result.metrics.roc_auc > 0.5


@pytest.mark.unit
def test_unknown_model_raises(dataset) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        train(dataset, TrainConfig(model="mystery"))


@pytest.mark.unit
def test_fairness_metrics_shape() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    prob = rng.uniform(size=1000)
    groups = np.where(rng.uniform(size=1000) < 0.5, "A", "B")
    fm = compute_fairness(y, prob, groups, "group")
    assert set(fm.selection_rate.keys()) == {"A", "B"}
    assert fm.tpr_gap >= 0.0


@pytest.mark.unit
def test_metrics_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, prob)
    assert m.roc_auc == 1.0
