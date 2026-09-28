"""Generic scikit-learn model adapter (model-agnostic).

Wraps ANY scikit-learn estimator or ``Pipeline`` (preprocessing + estimator is one
deployable model). It takes a feature-vector dict ``{name: value}``, orders the values
by the model's known feature names, and calls the estimator. No domain-specific
transformation is applied - a Pipeline's own preprocessing runs inside ``predict``.

The generic core only ever sees :class:`PredictionResult` / :class:`ModelMetadata` /
:class:`FeatureSchema`, never the sklearn object.

sklearn is imported lazily so this module can be imported for type/interface checks
without sklearn present.
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

FRAMEWORK = "scikit-learn"


def detect_feature_names(model: Any, fallback: list[str] | None = None) -> list[str]:
    """Best-effort detection of the model's expected input feature names.

    Uses sklearn's ``feature_names_in_`` (set when fit on a DataFrame), descending into
    a Pipeline's first step if needed, then falls back to the provided names (e.g. the
    reference CSV columns).
    """
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    steps = getattr(model, "steps", None)
    if steps:
        first = steps[0][1]
        names = getattr(first, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]
    return list(fallback or [])


def detect_task(model: Any) -> str:
    """Classify the estimator as 'classification' or 'regression' where possible.

    Uses sklearn's official ``is_classifier``/``is_regressor`` (which understand
    Pipelines and estimator tags across versions), with duck-typing fallbacks.
    """
    try:
        from sklearn.base import is_classifier, is_regressor

        if is_regressor(model):
            return "regression"
        if is_classifier(model):
            return "classification"
    except Exception:  # noqa: BLE001 - sklearn may be absent or API may differ
        pass

    est = model
    steps = getattr(model, "steps", None)
    if steps:
        est = steps[-1][1]
    if getattr(est, "_estimator_type", None) == "regressor":
        return "regression"
    if hasattr(model, "predict_proba") or hasattr(est, "classes_"):
        return "classification"
    return "unknown"


class SklearnAdapter(ModelAdapter):
    """Adapter for any scikit-learn estimator or Pipeline over a fixed feature set."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        *,
        model_id: str,
        model_version: str | None = None,
        name: str | None = None,
        task: str | None = None,
    ) -> None:
        self._model = model
        self._features = [str(f) for f in feature_names]
        self._model_id = model_id
        self._model_version = model_version
        self._name = name or type(model).__name__
        self._task = task or detect_task(model)

    # ---- introspection ---------------------------------------------------------------
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            framework=FRAMEWORK,
            name=self._name,
            task=self._task,
            model_version=self._model_version,
            model_id=self._model_id,
            supports_proba=self.supports_proba(),
            extra={
                "estimator": type(self._model).__name__,
                "is_pipeline": bool(getattr(self._model, "steps", None)),
            },
        )

    def feature_schema(self) -> FeatureSchema:
        # For a generic model every input feature is a user-supplied raw input.
        return FeatureSchema(features=list(self._features), raw_inputs=list(self._features))

    def supports_proba(self) -> bool:
        return hasattr(self._model, "predict_proba")

    def supports_decision_function(self) -> bool:
        return hasattr(self._model, "decision_function")

    @property
    def task(self) -> str:
        return self._task

    @property
    def feature_names(self) -> list[str]:
        return list(self._features)

    # ---- prediction ------------------------------------------------------------------
    def _to_frame(self, features: dict[str, float | str]) -> pd.DataFrame:
        missing = [f for f in self._features if f not in features]
        if missing:
            raise AdapterError(
                "Missing required feature values.",
                detail=f"prediction requires: {missing}",
            )
        row = {f: features[f] for f in self._features}
        return pd.DataFrame([row], columns=self._features)

    def predict_one(self, features: dict[str, float | str]) -> PredictionResult:
        x = self._to_frame(features)
        # Preserve categorical (string) values as-is; coerce numerics to float so the
        # standardized event carries the exact values fed to the model.
        used: dict[str, float | str] = {
            f: (features[f] if isinstance(features[f], str) else _coerce_float(features[f]))
            for f in self._features
        }

        if self._task == "regression":
            value = float(self._model.predict(x)[0])
            return PredictionResult(
                prediction=None, probability=None, features=used, output=value
            )

        # Classification (or unknown-but-has-predict).
        probability: float | None = None
        if self.supports_proba():
            proba = self._model.predict_proba(x)[0]
            # Positive-class probability for binary; max-class probability otherwise.
            probability = float(proba[1]) if len(proba) == 2 else float(max(proba))
        pred = self._model.predict(x)[0]
        prediction = _coerce_int(pred)
        return PredictionResult(
            prediction=prediction,
            probability=probability,
            features=used,
            output=float(prediction) if prediction is not None else None,
        )

    def validate(self) -> None:
        if not (hasattr(self._model, "predict") or hasattr(self._model, "predict_proba")):
            raise AdapterError(
                "Unsupported model.",
                detail="object exposes neither predict nor predict_proba",
            )
        if not self._features:
            raise AdapterError(
                "The model's feature names could not be determined.",
                detail="no feature_names_in_ and no reference columns supplied",
            )

    # ---- escape hatch for framework-specific consumers (e.g. SHAP/LIME) --------------
    @property
    def raw_model(self) -> Any:
        """The underlying sklearn estimator/Pipeline (for explainers that need it)."""
        return self._model


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
