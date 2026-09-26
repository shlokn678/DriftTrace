"""Model explainability: SHAP (primary) + LIME (secondary) (FR-14).

Explanations use the actual deployed model and run OFF the prediction hot path
(FR-14.4): they are invoked from the ``/explain`` endpoint or the CLI, never inside
``POST /predict``. Output is structured and serializable and preserves the model
version context (FR-14.2).

Caveat carried in the output: feature attributions describe the model's local/global
behaviour; they are NOT proof of a causal mechanism.

SHAP/LIME are heavy optional deps (the ``explain`` extra) and are imported lazily so
the core library stays importable without them (NFR-8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from drifttrace.features.transform import MODEL_FEATURES

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


def _prepare_row(income: float, transform_params: Any) -> pd.DataFrame:
    from drifttrace.features.transform import transform

    frame = pd.DataFrame([{"income": float(income)}])
    featured = transform(frame, transform_params)
    return featured[MODEL_FEATURES]


def shap_explain_local(
    model: Any,
    income: float,
    transform_params: Any,
    *,
    model_version: str | None,
    background: pd.DataFrame | None = None,
) -> Explanation:
    """SHAP local explanation for a single prediction (primary explainer)."""
    import shap

    row = _prepare_row(income, transform_params)
    # A small deterministic background sample keeps KernelExplainer stable/fast.
    if background is None:
        background = row
    explainer = shap.Explainer(model.predict_proba, background)
    values = explainer(row)
    # For binary classifiers SHAP returns per-class; take the positive class.
    arr = np.array(values.values)
    if arr.ndim == 3:
        contrib = arr[0, :, 1]
        base = float(np.array(values.base_values)[0, 1])
    else:
        contrib = arr[0]
        base = float(np.ravel(values.base_values)[0])

    attributions = [
        Attribution(feature=f, value=float(row[f].iloc[0]), attribution=float(contrib[i]))
        for i, f in enumerate(MODEL_FEATURES)
    ]
    return Explanation(
        method="shap",
        scope="local",
        model_version=model_version,
        features=list(MODEL_FEATURES),
        attributions=attributions,
        base_value=base,
    )


def shap_global_importance(
    model: Any,
    sample: pd.DataFrame,
    *,
    model_version: str | None,
) -> Explanation:
    """SHAP global feature importance (mean absolute attribution over a sample)."""
    import shap

    X = sample[MODEL_FEATURES]
    explainer = shap.Explainer(model.predict_proba, X)
    values = explainer(X)
    arr = np.array(values.values)
    contrib = np.abs(arr[:, :, 1]).mean(axis=0) if arr.ndim == 3 else np.abs(arr).mean(axis=0)
    attributions = [
        Attribution(feature=f, value=float(X[f].mean()), attribution=float(contrib[i]))
        for i, f in enumerate(MODEL_FEATURES)
    ]
    attributions.sort(key=lambda a: abs(a.attribution), reverse=True)
    return Explanation(
        method="shap",
        scope="global",
        model_version=model_version,
        features=list(MODEL_FEATURES),
        attributions=attributions,
    )


def lime_explain_local(
    model: Any,
    income: float,
    transform_params: Any,
    training_sample: pd.DataFrame,
    *,
    model_version: str | None,
) -> Explanation:
    """LIME local explanation for a single prediction (secondary explainer)."""
    from lime.lime_tabular import LimeTabularExplainer

    row = _prepare_row(income, transform_params)
    X_train = training_sample[MODEL_FEATURES].to_numpy(dtype=float)
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(MODEL_FEATURES),
        class_names=["ok", "default"],
        discretize_continuous=True,
        random_state=0,
    )
    exp = explainer.explain_instance(
        row.to_numpy(dtype=float)[0],
        model.predict_proba,
        num_features=len(MODEL_FEATURES),
    )
    weights = dict(exp.as_map()[1])  # positive class
    attributions = [
        Attribution(
            feature=f,
            value=float(row[f].iloc[0]),
            attribution=float(weights.get(i, 0.0)),
        )
        for i, f in enumerate(MODEL_FEATURES)
    ]
    return Explanation(
        method="lime",
        scope="local",
        model_version=model_version,
        features=list(MODEL_FEATURES),
        attributions=attributions,
    )
