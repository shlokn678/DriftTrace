"""Deterministic drift-injection harness (FR-18) and demo/replay data.

Produces deterministic prediction-event streams for four scenarios so the REAL Phase 4
KS/PSI detectors and RCA engine can be exercised end to end. The injected scenarios
change feature VALUES only; they never hard-code a detector verdict - actual KS/PSI on
the events against the training baseline determine what drifts (FR-18.2).

Scenarios:
- control      : income from the training-like distribution -> expected no drift.
- income_annual: the pitch failure, income x12 (monthly->annual). Because credit_score
                 and risk_score are DERIVED from income, they move too -> income is the
                 earliest/root cause, credit_score/risk_score are downstream symptoms.
- mid_chain    : income normal, but credit_score is shifted directly (and risk follows)
                 -> credit_score is the earliest supported root, risk_score a symptom.
- two_roots    : income shifted AND risk_score independently perturbed, with credit_score
                 kept near-baseline -> two independent roots (income and risk_score).

The existing trained model remains the model of record; no second model is invented.
Same PredictionEvent schema as the live system, usable with FileReplaySource and the
Redpanda producer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drifttrace.features.transform import (
    MODEL_FEATURES,
    TransformParams,
    compute_credit_score,
    compute_risk_score,
    fit_params,
)
from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.source import FileSink

# Scenario names.
CONTROL = "control"
INCOME_ANNUAL = "income_annual"
MID_CHAIN = "mid_chain"
TWO_ROOTS = "two_roots"

# Backward-compatible aliases (Phase 3 tests/docs used these).
NORMAL = CONTROL
DRIFT = INCOME_ANNUAL

SCENARIOS = [CONTROL, INCOME_ANNUAL, MID_CHAIN, TWO_ROOTS]

_CREDIT_MIN = 300.0
_CREDIT_MAX = 900.0


def _base_income(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=8.5, sigma=0.5, size=n)


def _events(
    income: np.ndarray,
    credit: np.ndarray,
    risk: np.ndarray,
    scenario: str,
    model_version: str | None,
) -> list[PredictionEvent]:
    prediction = (risk >= 0.5).astype(int)
    events: list[PredictionEvent] = []
    for i in range(len(income)):
        features = {
            "income": float(round(float(income[i]), 6)),
            "credit_score": float(round(float(credit[i]), 6)),
            "risk_score": float(round(float(risk[i]), 6)),
        }
        events.append(
            PredictionEvent(
                schema_version="1.0",
                event_id=f"{scenario}-{i:05d}",
                request_id=f"{scenario}-req-{i:05d}",
                model_version=model_version,
                timestamp="2024-01-01T00:00:00+00:00",
                features={k: features[k] for k in MODEL_FEATURES},
                prediction=int(prediction[i]),
                probability=float(round(float(risk[i]), 6)),
                source=f"replay:{scenario}",
            )
        )
    return events


def generate_events(
    scenario: str,
    n: int = 200,
    seed: int = 7,
    model_version: str | None = None,
) -> list[PredictionEvent]:
    """Generate deterministic events for a scenario. Values only; no verdict is forced."""
    income = _base_income(n, seed)
    params: TransformParams = fit_params(income)
    rng = np.random.default_rng(seed + 1000)

    if scenario == CONTROL:
        credit = compute_credit_score(income, params)
        risk = compute_risk_score(credit, params)

    elif scenario == INCOME_ANNUAL:
        # Monthly -> annual: income x12. credit/risk derived from the shifted income,
        # so downstream nodes move as symptoms.
        income = income * 12.0
        credit = compute_credit_score(income, params)
        risk = compute_risk_score(credit, params)

    elif scenario == MID_CHAIN:
        # income stays baseline; credit_score is shifted directly downward, risk derived
        # from the shifted credit so risk moves as a symptom of credit.
        credit = compute_credit_score(income, params)
        credit = np.clip(credit - 150.0, _CREDIT_MIN, _CREDIT_MAX)
        risk = compute_risk_score(credit, params)

    elif scenario == TWO_ROOTS:
        # Two independent roots: income shifted (root 1) and risk_score independently
        # perturbed (root 2), while credit_score is kept near its baseline so it is not
        # itself a drifted intermediate that would make risk a mere symptom.
        base_income = income.copy()
        income = income * 12.0
        credit = compute_credit_score(base_income, params)  # credit from UNSHIFTED income
        risk = compute_risk_score(credit, params)
        # Independently push risk toward 1.0 (an exogenous shock unrelated to credit).
        risk = np.clip(risk + rng.uniform(0.35, 0.55, size=len(risk)), 0.0, 1.0)

    else:
        raise ValueError(f"unknown scenario '{scenario}' (expected one of {SCENARIOS})")

    return _events(income, credit, risk, scenario, model_version)


def write_events(events: list[PredictionEvent], path: str | Path) -> Path:
    """Write events to a JSON-lines file (deterministic)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    sink = FileSink(out)
    for ev in events:
        sink.emit(ev)
    return out


def write_demo_datasets(out_dir: str | Path, n: int = 200, seed: int = 7) -> dict[str, Path]:
    """Write all four scenario event files. Returns their paths."""
    out = Path(out_dir)
    paths: dict[str, Path] = {}
    for scenario in SCENARIOS:
        paths[scenario] = write_events(
            generate_events(scenario, n=n, seed=seed), out / f"events_{scenario}.jsonl"
        )
    return paths
