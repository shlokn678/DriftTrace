"""Integration tests for MLflow tracking + model registry (FR-5).

Uses a temporary local file-based tracking store so no server or account is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.features.transform import MODEL_FEATURES, fit_params, transform
from drifttrace.training.evidence import LifecycleEvidence, dataset_version, new_run_id
from drifttrace.training.train import TrainConfig, train


@pytest.fixture
def tracking_uri(tmp_path):
    # MLflow 3.x requires a database backend; use a local SQLite file (decision D-6).
    db = (tmp_path / "mlflow.db").as_posix()
    return f"sqlite:///{db}"


@pytest.fixture(scope="module")
def dataset():
    return generate(GeneratorParams(n_rows=3000, seed=41))


@pytest.mark.integration
def test_log_and_register_passing_model(dataset, tracking_uri) -> None:
    from drifttrace.training.registry import load_model, log_training_run

    estimator, result = train(dataset, TrainConfig(seed=1, min_roc_auc=0.6))
    assert result.passed_gate

    evidence = LifecycleEvidence(
        dataset_version=dataset_version(b"data"),
        code_commit="testcommit",
        pipeline_execution_id=new_run_id(),
    )
    logged = log_training_run(estimator, result, evidence, tracking_uri=tracking_uri)
    assert logged.run_id
    assert logged.model_version is not None

    # Load the exact registered version (FR-5 AC-3).
    loaded = load_model(logged.model_version, tracking_uri=tracking_uri)
    tparams = fit_params(dataset["income"].to_numpy(dtype=float))
    featured = transform(dataset[["income"]].head(5), tparams)
    prob = loaded.predict_proba(featured[MODEL_FEATURES])[:, 1]
    assert prob.shape == (5,)
    assert np.all((prob >= 0) & (prob <= 1))


@pytest.mark.integration
def test_below_gate_model_not_registered(dataset, tracking_uri) -> None:
    from drifttrace.training.registry import log_training_run

    estimator, result = train(dataset, TrainConfig(seed=1, min_roc_auc=1.01))
    assert not result.passed_gate
    logged = log_training_run(estimator, result, _evidence(), tracking_uri=tracking_uri)
    assert logged.model_version is None  # logged but not registered


@pytest.mark.integration
def test_two_versions_increment(dataset, tracking_uri) -> None:
    from drifttrace.training.registry import latest_model_version, log_training_run

    estimator, result = train(dataset, TrainConfig(seed=1, min_roc_auc=0.6))
    v1 = log_training_run(estimator, result, _evidence(), tracking_uri=tracking_uri).model_version
    v2 = log_training_run(estimator, result, _evidence(), tracking_uri=tracking_uri).model_version
    assert int(v2) == int(v1) + 1
    assert latest_model_version(tracking_uri=tracking_uri) == v2


def _evidence() -> LifecycleEvidence:
    return LifecycleEvidence(
        dataset_version=dataset_version(b"data"),
        code_commit="testcommit",
        pipeline_execution_id=new_run_id(),
    )
