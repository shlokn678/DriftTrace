"""Model explainability: SHAP (primary) + LIME (secondary), model-agnostic.

Explanations use the actual active model and run OFF the prediction hot path: they are
invoked from ``/explain`` or the CLI, never inside ``POST /predict``. Output is
structured, serializable, and preserves the model-version context.

Caveat carried in the output: feature attributions describe the model's behaviour; they
are NOT proof of a causal mechanism, and are NOT the same as drift root-cause analysis.

SHAP/LIME are heavy optional deps (the ``explain`` extra), imported lazily so the core
stays importable without them (NFR-8). All entry points operate on a feature-vector row
and the model's own feature names - no domain assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_CAVEAT = "Feature attributions describe model behaviour, not a proven causal mechanism."


@dataclass
class Attribution:
    feature: str
    value: float
    attribution: float


@dataclass
class Explanation:
    """A structured explanation for a single prediction or global importance."""

    method: str  # "shap" | "lime"
    scope: str  # "local" | "global"
    model_version: str | None
    features: list[str]
    attributions: list[Attribution] = field(default_factory=list)
    base_value: float | None = None
    caveat: str = _CAVEAT

    def to_dict(self) -> dict:
        return asdict(self)


def _row_frame(features: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    row = {f: float(features.get(f, 0.0)) for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _predict_fn(model: Any) -> Any:
    """Return a scalar-output predict function for SHAP/LIME.

    Uses positive-class probability for binary classifiers, else the max-class
    probability, else plain ``predict`` for regressors.
    """
    if hasattr(model, "predict_proba"):

        def f(x: Any) -> np.ndarray:
            proba = np.asarray(model.predict_proba(x))
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba.max(axis=1)

        return f
    return model.predict


def shap_explain_local(
    model: Any,
    features: dict[str, float],
    feature_names: list[str],
    *,
    model_version: str | None,
    background: pd.DataFrame,
) -> Explanation:
    """SHAP local explanation for a single prediction (primary explainer)."""
    import shap

    row = _row_frame(features, feature_names)
    fn = _predict_fn(model)
    explainer = shap.Explainer(fn, background[feature_names])
    values = explainer(row)
    arr = np.array(values.values)
    contrib = arr[0] if arr.ndim == 2 else np.ravel(arr)[: len(feature_names)]
    try:
        base = float(np.ravel(values.base_values)[0])
    except Exception:  # noqa: BLE001
        base = None

    attributions = [
        Attribution(
            feature=f,
            value=float(row[f].iloc[0]),
            attribution=float(contrib[i]),
        )
        for i, f in enumerate(feature_names)
    ]
    return Explanation(
        method="shap",
        scope="local",
        model_version=model_version,
        features=list(feature_names),
        attributions=attributions,
        base_value=base,
    )


def shap_global_importance(
    model: Any,
    sample: pd.DataFrame,
    feature_names: list[str],
    *,
    model_version: str | None,
) -> Explanation:
    """SHAP global feature importance (mean absolute attribution over a sample)."""
    import shap

    X = sample[feature_names]
    fn = _predict_fn(model)
    explainer = shap.Explainer(fn, X)
    values = explainer(X)
    arr = np.array(values.values)
    contrib = np.abs(arr).mean(axis=0)
    attributions = [
        Attribution(feature=f, value=float(X[f].mean()), attribution=float(contrib[i]))
        for i, f in enumerate(feature_names)
    ]
    attributions.sort(key=lambda a: abs(a.attribution), reverse=True)
    return Explanation(
        method="shap",
        scope="global",
        model_version=model_version,
        features=list(feature_names),
        attributions=attributions,
    )


def lime_explain_local(
    model: Any,
    features: dict[str, float],
    feature_names: list[str],
    training_sample: pd.DataFrame,
    *,
    model_version: str | None,
) -> Explanation:
    """LIME local explanation for a single prediction (secondary explainer)."""
    from lime.lime_tabular import LimeTabularExplainer

    row = _row_frame(features, feature_names)
    X_train = training_sample[feature_names].to_numpy(dtype=float)
    is_classifier = hasattr(model, "predict_proba")
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(feature_names),
        mode="classification" if is_classifier else "regression",
        discretize_continuous=True,
        random_state=0,
    )
    if is_classifier:
        exp = explainer.explain_instance(
            row.to_numpy(dtype=float)[0],
            model.predict_proba,
            num_features=len(feature_names),
        )
        label = list(exp.as_map().keys())[0]
        weights = dict(exp.as_map()[label])
    else:
        exp = explainer.explain_instance(
            row.to_numpy(dtype=float)[0],
            model.predict,
            num_features=len(feature_names),
        )
        weights = dict(exp.as_map()[0])

    attributions = [
        Attribution(
            feature=f,
            value=float(row[f].iloc[0]),
            attribution=float(weights.get(i, 0.0)),
        )
        for i, f in enumerate(feature_names)
    ]
    return Explanation(
        method="lime",
        scope="local",
        model_version=model_version,
        features=list(feature_names),
        attributions=attributions,
    )
