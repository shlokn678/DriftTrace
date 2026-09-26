"""FastAPI prediction service (FR-7).

Endpoints (FR-7.5): ``GET /health``, ``GET /ready``, ``POST /predict``,
``GET /model-info``. (``/explain`` and ``/rca/latest`` are Phase 4 and intentionally
not implemented here.)

Behaviour:
- Loads a specific registered model version via the existing MLflow registry
  conventions (FR-7.3); does not change the training/MLflow lifecycle.
- Recomputes features with the shared transform so serving matches training (FR-3.3).
- Logs every request (features, prediction, model version, timestamp) (FR-7.4).
- Emits a :class:`PredictionEvent` after each successful prediction, fire-and-forget,
  so prediction latency never depends on the monitor (FR-7.2, NFR-2).

Configuration is environment-driven (NFR-4/NFR-7); see ``serving/config.py``. The API
is unauthenticated and intended for localhost / the internal Compose network (FR-7.6).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from drifttrace.features.transform import MODEL_FEATURES, TransformParams, transform
from drifttrace.serving.config import ServingSettings, get_serving_settings
from drifttrace.serving.metrics import METRICS
from drifttrace.serving.schemas import (
    ExplainRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from drifttrace.streaming.event import PredictionEvent, new_event_id
from drifttrace.streaming.source import EventSink
from drifttrace.training.registry import REGISTERED_MODEL_NAME

logger = logging.getLogger("drifttrace.serving")


class ModelHolder:
    """Holds the loaded model, its version, and the feature-transform params.

    Loading is lazy and tolerant: if no model is registered yet, the service still
    starts and reports non-ready (FR-7 AC-2) instead of crashing.
    """

    def __init__(self, settings: ServingSettings) -> None:
        self.settings = settings
        self.model: Any = None
        self.model_version: str | None = None
        self.transform_params: TransformParams | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.transform_params is not None

    def load(self) -> None:
        """Load the configured (or latest) registered model version (FR-7.3)."""
        from drifttrace.training.registry import latest_model_version, load_model

        uri = self.settings.mlflow_tracking_uri
        version = self.settings.model_version or latest_model_version(tracking_uri=uri)
        if version is None:
            logger.warning("no registered model version available; service not ready")
            return
        self.model = load_model(version, tracking_uri=uri)
        self.model_version = str(version)
        self.transform_params = self.settings.load_transform_params()
        logger.info("loaded model version %s", self.model_version)


def _make_sink(settings: ServingSettings) -> EventSink:
    """Build the event sink from settings (Redpanda if configured, else file)."""
    if settings.use_redpanda:
        from drifttrace.streaming.source import RedpandaSink

        return RedpandaSink(brokers=settings.redpanda_brokers, topic=settings.topic)
    from drifttrace.streaming.source import FileSink

    return FileSink(settings.event_log_path)


def predict_one(
    holder: ModelHolder,
    req: PredictRequest,
    sink: EventSink | None,
) -> PredictResponse:
    """Core prediction path: transform -> predict -> log -> emit event.

    Extracted from the route so it is unit-testable without an HTTP client.
    """
    assert holder.transform_params is not None and holder.model is not None
    frame = pd.DataFrame([{"income": req.income}])
    featured = transform(frame, holder.transform_params)
    X = featured[MODEL_FEATURES]

    probability = float(holder.model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= 0.5)
    features = {name: float(featured[name].iloc[0]) for name in MODEL_FEATURES}

    # Log the request (FR-7.4) and count it (metrics).
    METRICS.inc("drifttrace_prediction_requests_total")
    logger.info(
        "prediction request_id=%s model_version=%s prediction=%s probability=%.6f",
        req.request_id,
        holder.model_version,
        prediction,
        probability,
    )

    # Emit a prediction event, fire-and-forget (FR-7.2). Never fail the request on it.
    event_emitted = False
    if sink is not None:
        event = PredictionEvent(
            event_id=new_event_id(),
            request_id=req.request_id,
            model_version=holder.model_version,
            features=features,
            prediction=prediction,
            probability=probability,
            source="api",
        )
        try:
            sink.emit(event)
            event_emitted = True
            METRICS.inc("drifttrace_prediction_events_total")
        except Exception as exc:  # noqa: BLE001 - emission must not break serving
            logger.warning("event emission failed: %s", exc)

    return PredictResponse(
        request_id=req.request_id,
        model_version=holder.model_version,
        prediction=prediction,
        probability=probability,
        features=features,
        event_emitted=event_emitted,
    )


def create_app(settings: ServingSettings | None = None) -> Any:
    """Build the serving FastAPI app. Imports FastAPI lazily (NFR-8)."""
    from fastapi import FastAPI, HTTPException

    settings = settings or get_serving_settings()
    holder = ModelHolder(settings)
    sink = _make_sink(settings)

    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        # Attempt to load the model at startup; stay up (non-ready) if none exists.
        try:
            holder.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model load at startup failed: %s", exc)
        yield

    app = FastAPI(title="DriftTrace Prediction API", version="1.0", lifespan=lifespan)
    app.state.holder = holder
    app.state.sink = sink

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        if holder.ready:
            return ReadyResponse(
                ready=True, model_version=holder.model_version, detail="model loaded"
            )
        return ReadyResponse(ready=False, model_version=None, detail="model not loaded")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_version=holder.model_version,
            model_name=REGISTERED_MODEL_NAME,
            features=list(MODEL_FEATURES),
            loaded=holder.ready,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not holder.ready:
            # Try a lazy load in case the model was registered after startup.
            holder.load()
        if not holder.ready:
            raise HTTPException(status_code=503, detail="model not loaded")
        return predict_one(holder, req, sink)

    @app.get("/rca/latest")
    def rca_latest() -> dict:
        """Return the latest persisted monitoring/RCA report (FR-7.5, FR-12.1).

        Reads the ``latest_rca.json`` written by the monitor. Returns a not-available
        status (never an error) when no monitoring cycle has run yet.
        """
        import json

        latest = settings.reports_dir_path / "latest_rca.json"
        if not latest.exists():
            return {"available": False, "detail": "no monitoring report yet"}
        return {"available": True, "report": json.loads(latest.read_text(encoding="utf-8"))}

    @app.post("/explain")
    def explain(req: ExplainRequest) -> dict:
        """SHAP (primary) or LIME (secondary) explanation, off the hot path (FR-14)."""
        if not holder.ready:
            holder.load()
        if not holder.ready:
            raise HTTPException(status_code=503, detail="model not loaded")

        from drifttrace.explain.explainer import lime_explain_local, shap_explain_local

        # A small deterministic background/training sample derived from the transform.
        bg = _explain_background(holder)
        try:
            if req.method == "lime":
                exp = lime_explain_local(
                    holder.model,
                    req.income,
                    holder.transform_params,
                    bg,
                    model_version=holder.model_version,
                )
            else:
                exp = shap_explain_local(
                    holder.model,
                    req.income,
                    holder.transform_params,
                    model_version=holder.model_version,
                    background=bg,
                )
        except Exception as exc:  # noqa: BLE001 - explainability must not 500 the API
            raise HTTPException(status_code=500, detail=f"explanation failed: {exc}") from exc
        return exp.to_dict()

    @app.get("/metrics")
    def metrics() -> Any:
        """Prometheus-compatible metrics text (no Prometheus/Grafana installed)."""
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(METRICS.to_prometheus())

    return app


def _explain_background(holder: ModelHolder) -> pd.DataFrame:
    """Build a small deterministic background sample of model features for explainers."""
    import numpy as np

    assert holder.transform_params is not None
    rng = np.random.default_rng(0)
    incomes = rng.lognormal(mean=8.5, sigma=0.5, size=50)
    frame = pd.DataFrame({"income": incomes})
    featured = transform(frame, holder.transform_params)
    return featured[MODEL_FEATURES]
