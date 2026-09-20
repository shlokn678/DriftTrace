"""Model training and evaluation (FR-4).

Deterministic split, fit, metrics, an evaluation gate on ROC-AUC, and persistence of
a single loadable pipeline artifact (FR-4.1..FR-4.5). Uses the shared feature
transform (FR-3.3) so training features are computed exactly as they are at serving.

The MVP model is a scikit-learn classifier (decision D-4): a logistic-regression
baseline and one gradient-boosted-tree candidate. Determinism (NFR-12) comes from
fixed seeds and a fixed split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drifttrace.config import DEFAULT_SEED
from drifttrace.features.transform import MODEL_FEATURES, TransformParams, fit_params, transform
from drifttrace.training.evaluate import (
    FairnessMetrics,
    Metrics,
    compute_fairness,
    compute_metrics,
)

TARGET = "default"
SENSITIVE_ATTR = "group"


@dataclass
class SplitData:
    """Deterministic train/val/test split (FR-4.2)."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    groups_test: np.ndarray


@dataclass
class TrainConfig:
    """Training configuration."""

    model: str = "gradient_boosting"  # "logistic_regression" | "gradient_boosting"
    seed: int = DEFAULT_SEED
    test_size: float = 0.2
    val_size: float = 0.2
    min_roc_auc: float = 0.7  # evaluation gate (FR-4.3)


@dataclass
class TrainResult:
    """Outcome of a training run."""

    passed_gate: bool
    metrics: Metrics
    fairness: FairnessMetrics
    transform_params: TransformParams
    config: TrainConfig
    feature_names: list[str] = field(default_factory=lambda: list(MODEL_FEATURES))

    def to_dict(self) -> dict:
        return {
            "passed_gate": self.passed_gate,
            "metrics": self.metrics.to_dict(),
            "fairness": self.fairness.to_dict(),
            "config": {
                "model": self.config.model,
                "seed": self.config.seed,
                "test_size": self.config.test_size,
                "val_size": self.config.val_size,
                "min_roc_auc": self.config.min_roc_auc,
            },
            "transform_params": {
                "income_ref_log_mean": self.transform_params.income_ref_log_mean,
                "income_ref_log_std": self.transform_params.income_ref_log_std,
                "credit_coef": self.transform_params.credit_coef,
                "risk_credit_coef": self.transform_params.risk_credit_coef,
            },
            "feature_names": self.feature_names,
        }


def deterministic_split(frame: pd.DataFrame, config: TrainConfig) -> SplitData:
    """Split into train/val/test deterministically and reproducibly (FR-4.2)."""
    y = frame[TARGET].to_numpy(dtype=int)
    groups = frame[SENSITIVE_ATTR].to_numpy()
    X = frame.drop(columns=[TARGET])

    X_tmp, X_test, y_tmp, y_test, _, g_test = train_test_split(
        X, y, groups, test_size=config.test_size, random_state=config.seed, stratify=y
    )
    val_ratio = config.val_size / (1.0 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, random_state=config.seed, stratify=y_tmp
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, g_test)


def _make_estimator(config: TrainConfig) -> Pipeline:
    if config.model == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=config.seed)
    elif config.model == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=config.seed)
    else:
        raise ValueError(f"unknown model '{config.model}'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train(frame: pd.DataFrame, config: TrainConfig | None = None) -> tuple[Pipeline, TrainResult]:
    """Train and evaluate a model. Returns the fitted pipeline and the result.

    The pipeline expects raw records with at least ``income``; it recomputes derived
    features via the shared transform before predicting, so the same object is used
    unchanged at serving time.
    """
    cfg = config or TrainConfig()

    # Fit the feature transform on the full income column, then apply it.
    tparams = fit_params(frame["income"].to_numpy(dtype=float))
    featured = transform(frame, tparams)

    split = deterministic_split(featured, cfg)
    estimator = _make_estimator(cfg)
    estimator.fit(split.X_train[MODEL_FEATURES], split.y_train)

    test_prob = estimator.predict_proba(split.X_test[MODEL_FEATURES])[:, 1]
    metrics = compute_metrics(split.y_test, test_prob)
    fairness = compute_fairness(split.y_test, test_prob, split.groups_test, SENSITIVE_ATTR)

    passed = metrics.roc_auc >= cfg.min_roc_auc
    result = TrainResult(
        passed_gate=passed,
        metrics=metrics,
        fairness=fairness,
        transform_params=tparams,
        config=cfg,
    )
    return estimator, result
