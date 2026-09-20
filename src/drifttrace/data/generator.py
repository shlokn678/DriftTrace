"""Deterministic, seeded synthetic loan-default dataset generator.

Implements FR-1.4: a local generator that materialises the pitch's chained features
``income -> credit_score -> risk_score`` plus a ``default`` target. No external
dataset is downloaded and no external API is called.

Causal structure (matches the declared dependency graph in config/graph.yaml):

    income        raw input, log-normal (monthly income units)
    credit_score  derived from income (+ noise), scaled to the 300-900 band
    risk_score    derived from credit_score (+ noise), squashed to [0, 1]
    default       Bernoulli target driven by risk_score (higher risk -> more defaults)

A ``group`` column is included as the declared sensitive attribute for fairness
checks (FR-15.4). It is generated independently of the causal chain so the base
dataset is fair by construction; drift/bias can be injected later by the harness.

Determinism (NFR-12): identical ``seed`` + ``n_rows`` produce byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from drifttrace.config import DEFAULT_SEED

# Column order is fixed so CSV output is byte-stable across runs.
COLUMNS: list[str] = ["income", "credit_score", "risk_score", "group", "default"]

# Credit-score band (matches config/schema.yaml).
CREDIT_MIN = 300.0
CREDIT_MAX = 900.0


@dataclass(frozen=True)
class GeneratorParams:
    """Parameters controlling the synthetic data-generating process.

    Defaults are chosen to produce a realistic, moderately imbalanced loan-default
    dataset. All values are recorded with the run for reproducibility.
    """

    n_rows: int = 5000
    seed: int = DEFAULT_SEED
    # income ~ LogNormal(mean_log, sigma_log) in monthly units
    income_mean_log: float = 8.5
    income_sigma_log: float = 0.5
    # noise scales for the derived features
    credit_noise: float = 40.0
    risk_noise: float = 0.35
    # fraction of rows in the minority sensitive group
    group_minority_fraction: float = 0.35


def _standardize(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std standardization used to chain features stably."""
    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0:
        return values - mean
    return (values - mean) / std


def generate(params: GeneratorParams | None = None) -> pd.DataFrame:
    """Generate the synthetic loan-default dataset as a :class:`pandas.DataFrame`.

    The columns are ordered per :data:`COLUMNS`. The function is pure: given the
    same params it returns an equal frame (NFR-12).
    """
    p = params or GeneratorParams()
    rng = np.random.default_rng(p.seed)

    # 1. income: raw input (monthly units), strictly positive.
    income = rng.lognormal(mean=p.income_mean_log, sigma=p.income_sigma_log, size=p.n_rows)

    # 2. credit_score: derived from income. Higher income -> higher credit score.
    #    Map standardized income into the credit band, add noise, then clip.
    income_z = _standardize(income)
    credit_center = (CREDIT_MIN + CREDIT_MAX) / 2.0
    credit_spread = (CREDIT_MAX - CREDIT_MIN) / 6.0  # ~3 sigma inside the band
    credit_score = credit_center + credit_spread * income_z
    credit_score = credit_score + rng.normal(0.0, p.credit_noise, size=p.n_rows)
    credit_score = np.clip(credit_score, CREDIT_MIN, CREDIT_MAX)

    # 3. risk_score: derived from credit_score. Lower credit -> higher risk.
    #    Squash a standardized, sign-flipped credit score through a logistic to [0, 1].
    credit_z = _standardize(credit_score)
    risk_logit = -1.5 * credit_z + rng.normal(0.0, p.risk_noise, size=p.n_rows)
    risk_score = 1.0 / (1.0 + np.exp(-risk_logit))

    # 4. default: Bernoulli target driven by risk_score.
    default = (rng.uniform(0.0, 1.0, size=p.n_rows) < risk_score).astype(int)

    # Sensitive attribute, independent of the causal chain (fair by construction).
    group = np.where(
        rng.uniform(0.0, 1.0, size=p.n_rows) < p.group_minority_fraction,
        "B",
        "A",
    )

    frame = pd.DataFrame(
        {
            "income": np.round(income, 6),
            "credit_score": np.round(credit_score, 6),
            "risk_score": np.round(risk_score, 6),
            "group": group,
            "default": default,
        },
        columns=COLUMNS,
    )
    return frame


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize the frame to deterministic CSV bytes (stable line endings)."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
