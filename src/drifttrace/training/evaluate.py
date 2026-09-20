"""Model evaluation metrics and fairness hooks (FR-4.4, FR-15.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    """Classification metrics (FR-4.4). ROC-AUC is the primary gate metric."""

    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FairnessMetrics:
    """Group fairness metrics across a declared sensitive attribute (FR-15.4)."""

    attribute: str
    selection_rate: dict[str, float] = field(default_factory=dict)
    tpr: dict[str, float] = field(default_factory=dict)
    fpr: dict[str, float] = field(default_factory=dict)
    selection_rate_gap: float = 0.0
    tpr_gap: float = 0.0
    fpr_gap: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    """Compute classification metrics from true labels and predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def _rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def compute_fairness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    attribute: str,
    threshold: float = 0.5,
) -> FairnessMetrics:
    """Compute per-group selection rate and TPR/FPR, plus the max gap across groups."""
    y_pred = (y_prob >= threshold).astype(int)
    fm = FairnessMetrics(attribute=attribute)

    for g in sorted({str(x) for x in groups}):
        gmask = groups.astype(str) == g
        fm.selection_rate[g] = _rate(y_pred[gmask] == 1)
        pos = gmask & (y_true == 1)
        neg = gmask & (y_true == 0)
        fm.tpr[g] = _rate(y_pred[pos] == 1)
        fm.fpr[g] = _rate(y_pred[neg] == 1)

    def _gap(d: dict[str, float]) -> float:
        return float(max(d.values()) - min(d.values())) if len(d) > 1 else 0.0

    fm.selection_rate_gap = _gap(fm.selection_rate)
    fm.tpr_gap = _gap(fm.tpr)
    fm.fpr_gap = _gap(fm.fpr)
    return fm
