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
from drifttrace.serving.schemas import (
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

    # Log the request (FR-7.4).
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

    return app
