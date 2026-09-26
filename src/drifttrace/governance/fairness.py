"""Fairness metrics using the DECLARED sensitive attribute (FR-15.4).

The sensitive attribute is read from ``config/governance.yaml`` and is never inferred.
Reuses the group-metric computation from ``training/evaluate.py`` (selection rate,
TPR/FPR gaps). Deterministic given the same data and model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from drifttrace.features.transform import MODEL_FEATURES, TransformParams, transform
from drifttrace.governance.config import GovernanceConfig
from drifttrace.training.evaluate import compute_fairness


@dataclass
class FairnessEvidence:
    """Serializable fairness evidence for the governance record."""

    sensitive_attribute: str
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_fairness(
    model: Any,
    frame: pd.DataFrame,
    transform_params: TransformParams,
    config: GovernanceConfig,
    *,
    target_col: str = "default",
) -> FairnessEvidence:
    """Compute group fairness metrics for the deployed model on ``frame``.

    Requires the declared sensitive attribute and the target column to be present.
    """
    if config.sensitive_attribute is None:
        raise ValueError("no sensitive_attribute declared in governance config")
    attr = config.sensitive_attribute
    if attr not in frame.columns:
        raise ValueError(f"declared sensitive attribute '{attr}' not present in data")
    if target_col not in frame.columns:
        raise ValueError(f"target column '{target_col}' not present in data")

    featured = transform(frame, transform_params)
    X = featured[MODEL_FEATURES]
    y_true = frame[target_col].to_numpy(dtype=int)
    groups = frame[attr].to_numpy()
    y_prob = model.predict_proba(X)[:, 1]

    fm = compute_fairness(y_true, np.asarray(y_prob), groups, attr)
    return FairnessEvidence(sensitive_attribute=attr, metrics=fm.to_dict())
