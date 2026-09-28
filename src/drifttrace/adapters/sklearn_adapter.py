"""scikit-learn model adapter (Phase 5) - the first supported adapter.

Wraps the existing DriftTrace loan model: a scikit-learn ``Pipeline`` plus the shared
feature transform (``TransformParams``). It computes the chained features
``income -> credit_score -> risk_score`` exactly as training does, then calls the
sklearn model. The generic core only ever sees :class:`PredictionResult` /
:class:`ModelMetadata` / :class:`FeatureSchema`, never the sklearn object.

sklearn is imported lazily so this module can be imported for type/interface checks
without sklearn present (though the loan model does require it at runtime).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from drifttrace.adapters.base import (
    AdapterError,
    FeatureSchema,
    ModelAdapter,
    ModelMetadata,
    PredictionResult,
)
from drifttrace.features.transform import (
    MODEL_FEATURES,
    RAW_INPUTS,
    TransformParams,
    transform,
)

FRAMEWORK = "scikit-learn"


class SklearnAdapter(ModelAdapter):
    """Adapter for a scikit-learn estimator that consumes the DriftTrace features.

    ``model`` is any object exposing ``predict``/``predict_proba`` (an sklearn
    Pipeline in the loan demo). ``transform_params`` drives the deterministic feature
    chain shared with training.
    """

    def __init__(
        self,
        model: Any,
        transform_params: TransformParams,
        *,
        model_id: str = "drifttrace-loan-default",
        model_version: str | None = None,
        name: str | None = "drifttrace-loan-default",
        task: str | None = "binary_classification",
    ) -> None:
        self._model = model
        self._params = transform_params
        self._model_id = model_id
        self._model_version = model_version
        self._name = name
        self._task = task

    # ---- introspection ---------------------------------------------------------------
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            framework=FRAMEWORK,
            name=self._name,
            task=self._task,
            model_version=self._model_version,
            model_id=self._model_id,
            supports_proba=self.supports_proba(),
            extra={"estimator": type(self._model).__name__},
        )

    def feature_schema(self) -> FeatureSchema:
        return FeatureSchema(features=list(MODEL_FEATURES), raw_inputs=list(RAW_INPUTS))

    def supports_proba(self) -> bool:
        return hasattr(self._model, "predict_proba")

    # ---- prediction ------------------------------------------------------------------
    def predict_one(self, raw_inputs: dict[str, float]) -> PredictionResult:
        if "income" not in raw_inputs:
            raise AdapterError(
                "Missing required input.",
                detail="the loan model requires an 'income' raw input",
            )
        frame = pd.DataFrame([{"income": float(raw_inputs["income"])}])
        featured = transform(frame, self._params)
        x = featured[MODEL_FEATURES]

        if self.supports_proba():
            probability = float(self._model.predict_proba(x)[:, 1][0])
            prediction = int(probability >= 0.5)
        else:
            probability = None
            prediction = int(self._model.predict(x)[0])

        features = {name: float(featured[name].iloc[0]) for name in MODEL_FEATURES}
        return PredictionResult(prediction=prediction, probability=probability, features=features)

    def validate(self) -> None:
        if not (hasattr(self._model, "predict") or hasattr(self._model, "predict_proba")):
            raise AdapterError(
                "Unsupported model.",
                detail="object exposes neither predict nor predict_proba",
            )

    # ---- escape hatch for framework-specific consumers (e.g. SHAP/LIME) --------------
    @property
    def raw_model(self) -> Any:
        """The underlying sklearn estimator (for explainers that need the object)."""
        return self._model

    @property
    def transform_params(self) -> TransformParams:
        """The feature-transform params (for explainers and background generation)."""
        return self._params
