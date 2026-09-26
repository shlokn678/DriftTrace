"""Deterministic demo/replay data + injector for the Phase 3 infrastructure.

Purpose (Phase 3 ONLY): prove the infrastructure can carry both normal and
simulated-drift events through source -> monitor/window -> webhook. This is test/demo
input, NOT drift detection: no KS/PSI, no RCA, no scoring is performed here or in the
monitor. Phase 4 owns the actual drift algorithms.

The "simulated drift" scenario reuses the pitch's motivating failure (monthly->annual
``income``) purely as a way to make the drift-window events visibly differ from the
normal ones. The existing trained model remains the model of record; no second model
is invented.
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

NORMAL = "normal"
DRIFT = "drift"


def _base_income(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=8.5, sigma=0.5, size=n)


def _events_from_income(
    income: np.ndarray,
    params: TransformParams,
    source: str,
    model_version: str | None,
) -> list[PredictionEvent]:
    credit = compute_credit_score(income, params)
    risk = compute_risk_score(credit, params)
    prediction = (risk >= 0.5).astype(int)
    events: list[PredictionEvent] = []
    for i in range(len(income)):
        features = {
            "income": float(round(income[i], 6)),
            "credit_score": float(round(credit[i], 6)),
            "risk_score": float(round(risk[i], 6)),
        }
        # Deterministic event id + timestamp so replays are byte-reproducible.
        events.append(
            PredictionEvent(
                schema_version="1.0",
                event_id=f"{source}-{i:05d}",
                request_id=f"{source}-req-{i:05d}",
                model_version=model_version,
                timestamp="2024-01-01T00:00:00+00:00",
                features={k: features[k] for k in MODEL_FEATURES},
                prediction=int(prediction[i]),
                probability=float(round(risk[i], 6)),
                source=f"replay:{source}",
            )
        )
    return events


def generate_events(
    scenario: str,
    n: int = 200,
    seed: int = 7,
    model_version: str | None = None,
) -> list[PredictionEvent]:
    """Generate deterministic events for ``scenario`` in {"normal", "drift"}.

    - normal: income drawn from the training-like distribution.
    - drift:  the same income annualized (x12) - the pitch's monthly->annual failure,
      used here ONLY to make drift-window events differ from normal ones.
    """
    income = _base_income(n, seed)
    params = fit_params(income)
    if scenario == DRIFT:
        income = income * 12.0
    elif scenario != NORMAL:
        raise ValueError(f"unknown scenario '{scenario}' (expected 'normal' or 'drift')")
    return _events_from_income(income, params, scenario, model_version)


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
    """Write both the normal and simulated-drift event files. Returns their paths."""
    out = Path(out_dir)
    normal = write_events(generate_events(NORMAL, n=n, seed=seed), out / "events_normal.jsonl")
    drift = write_events(generate_events(DRIFT, n=n, seed=seed), out / "events_drift.jsonl")
    return {"normal": normal, "drift": drift}
