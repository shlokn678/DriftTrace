"""Unit tests for the model adapter layer (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from drifttrace.adapters.base import (
    AdapterError,
    FeatureSchema,
    ModelMetadata,
    PredictionResult,
    to_standard_event,
)
from drifttrace.adapters.registry import build_adapter, inspect_model_file
from drifttrace.adapters.sklearn_adapter import SklearnAdapter
from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.features.transform import MODEL_FEATURES, fit_params
from drifttrace.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def trained():
    frame = generate(GeneratorParams(n_rows=1500, seed=42))
    estimator, _ = train(frame, TrainConfig(seed=42, min_roc_auc=0.6))
    params = fit_params(frame["income"].to_numpy(dtype=float))
    return estimator, params


@pytest.fixture
def adapter(trained):
    estimator, params = trained
    return SklearnAdapter(estimator, params, model_version="1")


# ---- sklearn adapter -----------------------------------------------------------------
@pytest.mark.unit
def test_metadata_detected(adapter) -> None:
    meta = adapter.metadata()
    assert isinstance(meta, ModelMetadata)
    assert meta.framework == "scikit-learn"
    assert meta.task == "binary_classification"
    assert meta.model_version == "1"
    assert meta.supports_proba is True


@pytest.mark.unit
def test_feature_schema(adapter) -> None:
    schema = adapter.feature_schema()
    assert isinstance(schema, FeatureSchema)
    assert schema.features == MODEL_FEATURES
    assert schema.raw_inputs == ["income"]
    assert schema.n_features == 3


@pytest.mark.unit
def test_predict_one_returns_standard_result(adapter) -> None:
    result = adapter.predict_one({"income": 4200.0})
    assert isinstance(result, PredictionResult)
    assert result.prediction in (0, 1)
    assert result.probability is not None and 0.0 <= result.probability <= 1.0
    assert set(result.features) == set(MODEL_FEATURES)


@pytest.mark.unit
def test_predict_one_missing_input_raises(adapter) -> None:
    with pytest.raises(AdapterError):
        adapter.predict_one({"not_income": 1.0})


@pytest.mark.unit
def test_validate_ok(adapter) -> None:
    adapter.validate()  # must not raise


# ---- standardized event conversion ---------------------------------------------------
@pytest.mark.unit
def test_to_standard_event(adapter) -> None:
    result = adapter.predict_one({"income": 5000.0})
    event = to_standard_event(result, adapter.metadata(), request_id="r1", source="api")
    assert event.request_id == "r1"
    assert event.model_id == "drifttrace-loan-default"
    assert event.model_version == "1"
    assert event.prediction == result.prediction
    assert set(event.features) == set(MODEL_FEATURES)
    assert event.source == "api"


# ---- registry / build --------------------------------------------------------------
@pytest.mark.unit
def test_build_adapter_for_sklearn(trained) -> None:
    estimator, params = trained
    adapter = build_adapter(estimator, params, model_version="2")
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.metadata().model_version == "2"


@pytest.mark.unit
def test_build_adapter_unsupported_raises(trained) -> None:
    _, params = trained

    class _NotAModel:
        pass

    with pytest.raises(AdapterError, match="Unsupported model"):
        build_adapter(_NotAModel(), params)


# ---- inspection ----------------------------------------------------------------------
@pytest.mark.unit
def test_inspect_supported_model_file(trained, tmp_path) -> None:
    import cloudpickle

    estimator, _ = trained
    path = tmp_path / "model.pkl"
    with path.open("wb") as fh:
        cloudpickle.dump(estimator, fh)

    result = inspect_model_file(path, reference_available=True, graph_available=True)
    assert result.supported is True
    assert result.framework == "scikit-learn"
    assert result.n_features == 3
    assert result.missing == []


@pytest.mark.unit
def test_inspect_reports_missing_reference_and_deps(trained, tmp_path) -> None:
    import cloudpickle

    estimator, _ = trained
    path = tmp_path / "model.pkl"
    with path.open("wb") as fh:
        cloudpickle.dump(estimator, fh)

    result = inspect_model_file(path, reference_available=False, graph_available=False)
    assert result.supported is True
    assert "reference_data" in result.missing
    assert "dependencies" in result.missing
    assert result.message is not None  # limited root-cause tracing note


@pytest.mark.unit
def test_inspect_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not a real model")
    result = inspect_model_file(path, reference_available=True, graph_available=True)
    assert result.supported is False
    assert result.message is not None


@pytest.mark.unit
def test_inspect_non_model_pickle(tmp_path) -> None:
    import cloudpickle

    path = tmp_path / "thing.pkl"
    with path.open("wb") as fh:
        cloudpickle.dump({"just": "a dict"}, fh)
    result = inspect_model_file(path, reference_available=True, graph_available=True)
    assert result.supported is False


@pytest.mark.unit
def test_event_helper_handles_no_proba() -> None:
    result = PredictionResult(prediction=1, probability=None, features={"income": 1.0})
    meta = ModelMetadata(framework="scikit-learn", model_id="m", model_version="1")
    event = to_standard_event(result, meta)
    assert event.probability is None
    assert event.prediction == 1


@pytest.mark.unit
def test_event_features_are_floats() -> None:
    result = PredictionResult(prediction=0, probability=0.2, features={"income": np.float64(3.0)})
    meta = ModelMetadata(framework="scikit-learn")
    event = to_standard_event(result, meta)
    assert isinstance(event.features["income"], float)
