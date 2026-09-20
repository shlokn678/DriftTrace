"""Feature engineering for the declared chained features (FR-3).

This module is the SINGLE source of truth for how derived features are computed
from raw inputs. Both the training data pipeline and the serving path call the same
functions, so there is no train/serve skew (FR-3.3, FR-3 AC-3).

Declared chain (config/graph.yaml):
    income  ->  credit_score  ->  risk_score

``income`` is the raw input. ``credit_score`` and ``risk_score`` are derived. The
model then consumes the feature columns to predict ``default``.

The transform is deterministic and stateless given its coefficients, so calling it
on a single serving record yields exactly the value it would have inside a batch
(FR-3 AC-3). Standardization uses fixed reference statistics captured at fit time
rather than per-batch statistics, precisely to avoid batch-dependent serving skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Derived-feature columns produced by the transform, in declared order.
DERIVED_FEATURES: list[str] = ["credit_score", "risk_score"]
# The raw input the chain starts from.
RAW_INPUTS: list[str] = ["income"]
# Columns fed to the model (raw inputs that remain useful + derived features).
MODEL_FEATURES: list[str] = ["income", "credit_score", "risk_score"]

# Credit-score band (matches config/schema.yaml).
CREDIT_MIN = 300.0
CREDIT_MAX = 900.0
_CREDIT_CENTER = (CREDIT_MIN + CREDIT_MAX) / 2.0
_CREDIT_SPREAD = (CREDIT_MAX - CREDIT_MIN) / 6.0


@dataclass(frozen=True)
class TransformParams:
    """Fixed coefficients for the deterministic feature chain.

    ``income_ref_*`` are the reference statistics used to standardize income. They
    are part of the transform contract so serving standardizes identically to
    training regardless of the incoming batch (avoids train/serve skew).
    """

    income_ref_log_mean: float
    income_ref_log_std: float
    credit_coef: float = 1.0
    risk_credit_coef: float = -1.5


def _standardize_log_income(income: np.ndarray, params: TransformParams) -> np.ndarray:
    log_income = np.log(np.clip(income, 1e-9, None))
    std = params.income_ref_log_std if params.income_ref_log_std != 0.0 else 1.0
    return (log_income - params.income_ref_log_mean) / std


def compute_credit_score(income: np.ndarray, params: TransformParams) -> np.ndarray:
    """Derive ``credit_score`` from ``income`` (deterministic, no noise)."""
    income_z = _standardize_log_income(income, params)
    credit = _CREDIT_CENTER + _CREDIT_SPREAD * params.credit_coef * income_z
    return np.clip(credit, CREDIT_MIN, CREDIT_MAX)


def compute_risk_score(credit_score: np.ndarray, params: TransformParams) -> np.ndarray:
    """Derive ``risk_score`` from ``credit_score`` (deterministic, no noise)."""
    # Standardize credit within its band, flip sign (low credit -> high risk).
    credit_z = (credit_score - _CREDIT_CENTER) / _CREDIT_SPREAD
    risk_logit = params.risk_credit_coef * credit_z
    return 1.0 / (1.0 + np.exp(-risk_logit))


def fit_params(income: np.ndarray) -> TransformParams:
    """Fit the reference statistics used by the transform from training income."""
    log_income = np.log(np.clip(income, 1e-9, None))
    return TransformParams(
        income_ref_log_mean=float(log_income.mean()),
        income_ref_log_std=float(log_income.std()),
    )


def transform(frame: pd.DataFrame, params: TransformParams) -> pd.DataFrame:
    """Compute derived features from raw inputs.

    Requires an ``income`` column. Returns a copy of ``frame`` with ``credit_score``
    and ``risk_score`` (re)computed deterministically from ``income``. Any existing
    derived columns are overwritten so serving cannot smuggle in inconsistent values.
    """
    if "income" not in frame.columns:
        raise KeyError("transform requires an 'income' column")
    out = frame.copy()
    income = out["income"].to_numpy(dtype=float)
    credit = compute_credit_score(income, params)
    risk = compute_risk_score(credit, params)
    out["credit_score"] = credit
    out["risk_score"] = risk
    return out
