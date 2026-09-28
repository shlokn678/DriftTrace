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
import os
from typing import Any

import pandas as pd
from fastapi import File, UploadFile

from drifttrace.adapters.base import ModelAdapter, to_standard_event
from drifttrace.features.transform import MODEL_FEATURES, TransformParams, transform
from drifttrace.serving.config import ServingSettings, get_serving_settings
from drifttrace.serving.metrics import METRICS
from drifttrace.serving.schemas import (
    ExplainRequest,
    HealthResponse,
    ModelInfoResponse,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    RunScenarioRequest,
    UploadModelResponse,
)
from drifttrace.streaming.source import EventSink
from drifttrace.training.registry import REGISTERED_MODEL_NAME

logger = logging.getLogger("drifttrace.serving")


class ModelHolder:
    """Holds the loaded model behind a model-agnostic adapter (Phase 5).

    The serving core interacts with ``self.adapter`` (a ModelAdapter), never with a
    framework-specific object. Loading is lazy and tolerant: if no model is registered
    yet, the service still starts and reports non-ready (FR-7 AC-2) instead of crashing.

    ``model``/``transform_params``/``model_version`` remain available as compatibility
    accessors for the explainer, which still consumes the raw sklearn estimator.
    """

    def __init__(self, settings: ServingSettings) -> None:
        self.settings = settings
        self.adapter: ModelAdapter | None = None
        self.model_version: str | None = None

    @property
    def ready(self) -> bool:
        return self.adapter is not None

    @property
    def model(self) -> Any:
        """The underlying estimator, if the adapter exposes one (sklearn)."""
        return getattr(self.adapter, "raw_model", None)

    @property
    def transform_params(self) -> TransformParams | None:
        """The feature-transform params, if the adapter exposes them (sklearn)."""
        return getattr(self.adapter, "transform_params", None)

    def load(self) -> None:
        """Load the configured (or latest) registered model version behind an adapter."""
        from drifttrace.adapters.registry import build_adapter
        from drifttrace.training.registry import latest_model_version, load_model

        uri = self.settings.mlflow_tracking_uri
        version = self.settings.model_version or latest_model_version(tracking_uri=uri)
        if version is None:
            logger.warning("no registered model version available; service not ready")
            return
        estimator = load_model(version, tracking_uri=uri)
        params = self.settings.load_transform_params()
        self.model_version = str(version)
        # Wrap the loaded estimator in the model-agnostic adapter.
        self.adapter = build_adapter(estimator, params, model_version=str(version))
        logger.info("loaded model version %s via %s adapter", self.model_version, "sklearn")


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
    """Core prediction path: adapter.predict -> log -> emit standardized event.

    Model-agnostic: it calls the adapter, never a framework-specific object. Extracted
    from the route so it is unit-testable without an HTTP client.
    """
    assert holder.adapter is not None
    result = holder.adapter.predict_one({"income": req.income})
    prediction = result.prediction
    probability = result.probability
    features = result.features

    # Log the request (FR-7.4) and count it (metrics).
    METRICS.inc("drifttrace_prediction_requests_total")
    logger.info(
        "prediction request_id=%s model_version=%s prediction=%s probability=%s",
        req.request_id,
        holder.model_version,
        prediction,
        "n/a" if probability is None else f"{probability:.6f}",
    )

    # Emit a standardized prediction event, fire-and-forget (FR-7.2). Never fail the
    # request on it. Conversion goes through the single adapter->event boundary.
    event_emitted = False
    if sink is not None:
        event = to_standard_event(result, holder.adapter.metadata(), request_id=req.request_id)
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
        probability=probability if probability is not None else 0.0,
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

    # CORS for the local operations dashboard (dev + internal Compose network only).
    # Not a public API (FR-7.6). Allowed origins are configurable via env.
    from fastapi.middleware.cors import CORSMiddleware

    origins = settings.cors_allow_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.get("/demo/scenarios")
    def demo_scenarios() -> dict:
        """List the deterministic drift-injection scenarios (FR-18)."""
        from drifttrace.streaming.demo import SCENARIOS

        return {"scenarios": list(SCENARIOS)}

    @app.post("/demo/run-scenario")
    def demo_run_scenario(req: RunScenarioRequest) -> dict:
        """Run a deterministic scenario through the REAL Phase 4 pipeline (FR-18, FR-9/10/11).

        This is a thin HTTP wrapper over the exact same drift -> RCA -> report -> alert
        pipeline the CLI `replay` uses. It generates deterministic events (values only;
        the detector decides drift), runs the monitor windower + Phase4Processor, writes
        the monitoring report + latest_rca.json, and returns the outcome. No fabricated
        results: KS/PSI and RCA produce the verdicts.
        """
        return run_demo_scenario(settings, req.scenario, n=req.n, seed=req.seed)

    # ---- Phase 5: model onboarding ---------------------------------------------------
    @app.post("/models/upload", response_model=UploadModelResponse)
    async def upload_model(file: UploadFile = File(...)) -> UploadModelResponse:  # noqa: B008
        """Upload one model file; DriftTrace inspects it and reports what it found.

        Minimal input: the model file is the only required upload. Reference data and
        dependency information are reused from the project context when available and
        only requested when genuinely missing.
        """
        import tempfile
        from pathlib import Path as _Path

        from drifttrace.serving.onboarding import REGISTRY

        suffix = _Path(file.filename or "model.pkl").suffix or ".pkl"
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = _Path(tmp.name)
        try:
            onboarded = REGISTRY.onboard(tmp_path, original_name=file.filename)
        finally:
            tmp_path.unlink(missing_ok=True)

        return UploadModelResponse(
            model_id=onboarded.model_id if onboarded.supported else None,
            supported=onboarded.supported,
            framework=onboarded.framework,
            name=onboarded.name,
            task=onboarded.task,
            n_features=onboarded.n_features,
            supports_proba=onboarded.supports_proba,
            features=onboarded.features,
            reference_available=onboarded.reference_available,
            dependencies_available=onboarded.dependencies_available,
            missing=onboarded.missing,
            ready_to_monitor=onboarded.ready_to_monitor,
            message=onboarded.message,
        )

    @app.get("/models/{model_id}", response_model=ModelStatusResponse)
    def model_status(model_id: str) -> ModelStatusResponse:
        """Return the onboarding/registration status of a previously uploaded model."""
        from fastapi import HTTPException

        from drifttrace.serving.onboarding import REGISTRY

        onboarded = REGISTRY.get(model_id)
        if onboarded is None:
            raise HTTPException(status_code=404, detail="model not found")
        return ModelStatusResponse(
            model_id=onboarded.model_id,
            supported=onboarded.supported,
            framework=onboarded.framework,
            name=onboarded.name,
            task=onboarded.task,
            n_features=onboarded.n_features,
            reference_available=onboarded.reference_available,
            dependencies_available=onboarded.dependencies_available,
            ready_to_monitor=onboarded.ready_to_monitor,
            message=onboarded.message,
        )

    return app


def run_demo_scenario(
    settings: ServingSettings, scenario: str, *, n: int = 300, seed: int = 7
) -> dict:
    """Execute a deterministic scenario through the real Phase 4 pipeline."""
    from drifttrace.alerting.alerter import Alerter
    from drifttrace.alerting.webhook import WebhookClient
    from drifttrace.config import get_paths
    from drifttrace.drift.baseline import Baseline
    from drifttrace.drift.config import load_drift_config
    from drifttrace.graph.loader import load_graph
    from drifttrace.streaming.demo import SCENARIOS, generate_events
    from drifttrace.streaming.processing import run_scenario_over_events

    if scenario not in SCENARIOS:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=f"unknown scenario '{scenario}'")

    paths = get_paths()
    baseline = Baseline.load(paths.artifacts / "baseline.json")
    graph = load_graph()
    config = load_drift_config()

    webhook_url = os.environ.get("WEBHOOK_STUB_URL")
    webhook = WebhookClient(webhook_url) if webhook_url else None
    alerter = Alerter(
        webhook=webhook,
        cooldown_seconds=0,  # demo: always emit so the UI shows the alert
        alert_log_path=settings.reports_dir_path / "alerts.jsonl",
    )
    model_version = str(baseline.model_version) if baseline.model_version is not None else None
    events = generate_events(scenario, n=n, seed=seed, model_version=model_version)
    outcomes = run_scenario_over_events(
        events,
        baseline,
        graph,
        config,
        alerter=alerter,
        reports_dir=settings.reports_dir_path,
        window_size=n,
        min_window_samples=30,
    )
    outcome = outcomes[-1] if outcomes else None
    return {
        "scenario": scenario,
        "windows": len(outcomes),
        "outcome": outcome.to_dict() if outcome is not None else None,
    }


def _explain_background(holder: ModelHolder) -> pd.DataFrame:
    """Build a small deterministic background sample of model features for explainers."""
    import numpy as np

    assert holder.transform_params is not None
    rng = np.random.default_rng(0)
    incomes = rng.lognormal(mean=8.5, sigma=0.5, size=50)
    frame = pd.DataFrame({"income": incomes})
    featured = transform(frame, holder.transform_params)
    return featured[MODEL_FEATURES]
