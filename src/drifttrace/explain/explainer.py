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
class Factor:
    """A single interpreted contributor to a local explanation."""

    feature: str
    label: str  # human-readable feature label
    direction: str  # "toward_prediction" | "away_from_prediction"
    strength: str  # "strong" | "moderate" | "small"
    value: float  # the SHAP/LIME attribution value (signed)


@dataclass
class Interpretation:
    """Deterministic natural-language interpretation of a local explanation.

    Built purely from the attribution signs/magnitudes, the predicted output, and the
    method - no external model, no fabricated feature meanings. Describes *influence on
    the prediction*, never causation.
    """

    method: str  # "SHAP" | "LIME"
    scope_label: str  # e.g. "Local explanation for this prediction"
    prediction_label: str  # e.g. "Malignant" / "Class 1" / "Predicted value: 84.2"
    summary: str
    supporting_factors: list[Factor] = field(default_factory=list)
    opposing_factors: list[Factor] = field(default_factory=list)
    other_note: str | None = None
    caveat: str = _CAVEAT

    def to_dict(self) -> dict:
        return asdict(self)


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
    # Optional deterministic natural-language layer (local explanations only).
    interpretation: Interpretation | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def humanize_feature(name: str) -> str:
    """Turn a feature id into a readable label (underscores -> spaces, capitalized).

    Only reformats the given name; it never invents a definition or meaning.
    """
    cleaned = str(name).replace("_", " ").strip()
    if not cleaned:
        return str(name)
    return cleaned[0].upper() + cleaned[1:]


def _strength(magnitude: float, max_magnitude: float) -> str:
    if max_magnitude <= 0:
        return "small"
    ratio = magnitude / max_magnitude
    if ratio >= 0.66:
        return "strong"
    if ratio >= 0.33:
        return "moderate"
    return "small"


def _join(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def interpret_explanation(
    explanation: Explanation,
    *,
    prediction_label: str,
    task: str,
    top_k: int = 3,
    min_fraction: float = 0.05,
) -> Interpretation:
    """Build a deterministic natural-language interpretation of a local explanation.

    ``prediction_label`` is the class name (classification) or a formatted value string
    (regression). ``task`` is "classification" or "regression". The interpretation
    describes how features *pushed the prediction*, never causation.
    """
    method_label = explanation.method.upper()
    # Rank by absolute attribution; keep only non-negligible contributors.
    attrs = list(explanation.attributions)
    max_mag = max((abs(a.attribution) for a in attrs), default=0.0)
    threshold = max_mag * min_fraction
    ranked = sorted(
        (a for a in attrs if abs(a.attribution) > threshold),
        key=lambda a: abs(a.attribution),
        reverse=True,
    )

    def _factor(a: Attribution, toward: bool) -> Factor:
        return Factor(
            feature=a.feature,
            label=humanize_feature(a.feature),
            direction="toward_prediction" if toward else "away_from_prediction",
            strength=_strength(abs(a.attribution), max_mag),
            value=float(a.attribution),
        )

    # Positive attribution => pushed toward the explained output; negative => away.
    supporting = [_factor(a, True) for a in ranked if a.attribution > 0][:top_k]
    opposing = [_factor(a, False) for a in ranked if a.attribution < 0][:top_k]

    n_shown = len(supporting) + len(opposing)
    other_note = "Other features had smaller effects." if len(ranked) > n_shown else None

    # Build the 2-4 sentence summary.
    if task == "regression":
        sent1 = f"Predicted value: {prediction_label}."
        toward_phrase = "increased the predicted value relative to the explanation baseline"
        away_phrase = "pushed the predicted value lower"
    else:
        sent1 = f"Prediction: {prediction_label}."
        toward_phrase = f"pushed the prediction toward {prediction_label}"
        away_phrase = "pushed the prediction the other way"

    parts = [sent1]
    if supporting:
        names = _join([f.label for f in supporting])
        parts.append(f"The prediction was influenced most by {names}, which {toward_phrase}.")
    elif opposing:
        names = _join([f.label for f in opposing])
        parts.append(f"The strongest contributors were {names}, which {away_phrase}.")
    else:
        parts.append("No single feature had a notable influence on this prediction.")

    if supporting and opposing:
        parts.append(f"{_join([f.label for f in opposing])} {away_phrase}.")

    summary = " ".join(parts)

    return Interpretation(
        method=method_label,
        scope_label="Local explanation for this prediction",
        prediction_label=prediction_label,
        summary=summary,
        supporting_factors=supporting,
        opposing_factors=opposing,
        other_note=other_note,
    )


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
