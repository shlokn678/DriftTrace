"""Reference-data profiling (model-agnostic).

The reference CSV is the baseline distribution for one model. From it DriftTrace
derives, with no domain assumptions:

- the feature schema (column names + numeric/categorical kind),
- per-feature reference distributions (for KS/PSI drift),
- the sample count.

The profile is turned into the existing :class:`~drifttrace.drift.baseline.Baseline`
so the generic drift engine works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from drifttrace.drift.baseline import Baseline, NodeBaseline


class ReferenceError(ValueError):
    """Raised when the reference data is unusable. Carries a user-facing message."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# The canonical name for the model's output node in the dependency graph / baseline.
OUTPUT_NODE = "prediction"

# Minimum rows for the reference to be usable for drift detection.
MIN_REFERENCE_ROWS = 20


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype).startswith("category")


@dataclass
class FeatureProfile:
    """Profile of one reference feature column."""

    name: str
    kind: str  # "continuous" | "categorical"
    count: int
    values: list[float] = field(default_factory=list)  # continuous sample
    categories: dict[str, float] = field(default_factory=dict)  # category -> proportion


@dataclass
class ReferenceProfile:
    """The derived schema + baseline distributions for a model's reference data."""

    feature_names: list[str]
    features: dict[str, FeatureProfile]
    n_rows: int

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def numeric_features(self) -> list[str]:
        return [f for f in self.feature_names if self.features[f].kind == "continuous"]

    def categorical_features(self) -> list[str]:
        return [f for f in self.feature_names if self.features[f].kind == "categorical"]


def profile_reference(
    frame: pd.DataFrame,
    *,
    max_reference_samples: int = 5000,
) -> ReferenceProfile:
    """Profile a reference DataFrame into a feature schema + distributions.

    Every column is treated as a feature (the caller decides column selection). Numeric
    columns become continuous features; object/category columns become categorical.
    """
    import numpy as np

    if frame.shape[1] == 0:
        raise ReferenceError("Reference data has no columns.")
    if len(frame) < MIN_REFERENCE_ROWS:
        raise ReferenceError(
            f"Reference data has too few rows ({len(frame)}); "
            f"at least {MIN_REFERENCE_ROWS} are needed to establish a baseline."
        )

    features: dict[str, FeatureProfile] = {}
    names: list[str] = []
    for col in frame.columns:
        name = str(col)
        names.append(name)
        series = frame[col].dropna()
        if _is_categorical(series):
            counts = series.value_counts(normalize=True)
            features[name] = FeatureProfile(
                name=name,
                kind="categorical",
                count=int(series.size),
                categories={str(k): float(v) for k, v in counts.items()},
            )
        else:
            values = series.to_numpy(dtype=float)
            if values.size > max_reference_samples:
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(values.size, size=max_reference_samples, replace=False))
                values = values[idx]
            features[name] = FeatureProfile(
                name=name,
                kind="continuous",
                count=int(series.size),
                values=[float(v) for v in values],
            )
    return ReferenceProfile(feature_names=names, features=features, n_rows=len(frame))


def baseline_from_profile(
    profile: ReferenceProfile,
    *,
    model_version: str,
    output_values: list[float] | None = None,
) -> Baseline:
    """Build a drift :class:`Baseline` from a reference profile.

    Each feature becomes a node baseline. If ``output_values`` (the model's predictions
    over the reference) is provided, an ``OUTPUT_NODE`` continuous baseline is added so
    output drift can be monitored too.
    """
    baseline = Baseline(model_version=model_version)
    for name in profile.feature_names:
        fp = profile.features[name]
        if fp.kind == "categorical":
            baseline.nodes[name] = NodeBaseline(
                node=name,
                kind="categorical",
                categories=dict(fp.categories),
                count=fp.count,
            )
        else:
            baseline.nodes[name] = NodeBaseline(
                node=name,
                kind="continuous",
                values=list(fp.values),
                count=fp.count,
            )
    if output_values:
        baseline.nodes[OUTPUT_NODE] = NodeBaseline(
            node=OUTPUT_NODE,
            kind="continuous",
            values=[float(v) for v in output_values],
            count=len(output_values),
        )
    return baseline
