"""FastAPI prediction + monitoring service (model-agnostic).

The service serves whichever model the operator has uploaded and activated. There is no
built-in or default model: a fresh install starts with NO active model and reports
non-ready until a bundle is activated.

Flow:
- Upload a bundle (model.pkl + reference.csv + optional graph.json) -> inspect.
- Activate it ("use this model") -> it becomes the active model.
- ``POST /predict`` serves the active model with a generic feature-vector payload and
  emits a standardized prediction event (fire-and-forget), which feeds monitoring.
- Drift is measured against the active model's reference baseline; RCA uses its optional
  dependency graph.

Configuration is environment-driven (NFR-4/NFR-7); see ``serving/config.py``. The API is
unauthenticated and intended for localhost / internal use only.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import File, UploadFile

from drifttrace.adapters.base import to_standard_event
from drifttrace.serving.config import ServingSettings, get_serving_settings
from drifttrace.serving.metrics import METRICS
from drifttrace.serving.onboarding import REGISTRY, ActiveContext
from drifttrace.serving.schemas import (
    ActiveModelResponse,
    DriftTestRequest,
    ExplainRequest,
    HealthResponse,
    ModelInfoResponse,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    UploadModelResponse,
)
from drifttrace.streaming.source import EventSink

logger = logging.getLogger("drifttrace.serving")


def _active_ctx() -> ActiveContext | None:
    """The active model context (adapter + baseline + graph), or None if no active model."""
    return REGISTRY.active_context()


def _make_sink(settings: ServingSettings) -> EventSink:
    """Build the event sink from settings (optional Redpanda, else a local file)."""
    if settings.use_redpanda:
        from drifttrace.streaming.source import RedpandaSink

        return RedpandaSink(brokers=settings.redpanda_brokers, topic=settings.topic)
    from drifttrace.streaming.source import FileSink

    return FileSink(settings.event_log_path)


def predict_one(ctx: ActiveContext, req: PredictRequest, sink: EventSink | None) -> PredictResponse:
    """Core prediction path: active adapter -> log -> emit standardized event."""
    result = ctx.adapter.predict_one(dict(req.features))

    METRICS.inc("drifttrace_prediction_requests_total")
    logger.info(
        "prediction request_id=%s model=%s prediction=%s output=%s",
        req.request_id,
        ctx.model_id,
        result.prediction,
        result.output,
    )

    event_emitted = False
    if sink is not None:
        event = to_standard_event(result, ctx.adapter.metadata(), request_id=req.request_id)
        try:
            sink.emit(event)
            event_emitted = True
            METRICS.inc("drifttrace_prediction_events_total")
        except Exception as exc:  # noqa: BLE001 - emission must never break serving
            logger.warning("event emission failed: %s", exc)

    return PredictResponse(
        request_id=req.request_id,
        model_version=ctx.adapter.metadata().model_version,
        prediction=result.prediction,
        probability=result.probability,
        output=result.output,
        features=dict(result.features),
        event_emitted=event_emitted,
    )


def _prediction_label(raw_model: Any, result: Any, task: str) -> str:
    """Human-facing label for the explained prediction.

    Classification: the model's own class label (via ``classes_``) when available, else
    ``Class <n>``. Regression: a rounded predicted value string. No fabricated meanings.
    """
    if task == "regression":
        value = result.output if result.output is not None else 0.0
        return f"{float(value):.2f}"
    classes = getattr(raw_model, "classes_", None)
    steps = getattr(raw_model, "steps", None)
    if classes is None and steps:
        classes = getattr(steps[-1][1], "classes_", None)
    pred = result.prediction
    if classes is not None and pred is not None:
        try:
            label = classes[int(pred)]
        except (IndexError, ValueError, TypeError):
            label = None
        if label is not None:
            # A non-numeric class name is used as-is; a plain integer reads better as
            # "Class N" (we never invent a semantic name the model didn't declare).
            import numbers

            return f"Class {label}" if isinstance(label, numbers.Number) else str(label)
    return f"Class {pred}" if pred is not None else "prediction"


def _upload_response(onboarded: Any) -> UploadModelResponse:
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
        reference_rows=onboarded.reference_rows,
        missing=onboarded.missing,
        ready_to_monitor=onboarded.ready_to_monitor,
        message=onboarded.message,
    )


def create_app(settings: ServingSettings | None = None) -> Any:
    """Build the serving FastAPI app. Imports FastAPI lazily (NFR-8)."""
    from fastapi import FastAPI, HTTPException

    settings = settings or get_serving_settings()
    sink = _make_sink(settings)

    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        # No model is loaded at startup. The operator uploads + activates a bundle.
        yield

    app = FastAPI(title="DriftTrace API", version="2.0", lifespan=lifespan)
    app.state.sink = sink

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- health / readiness / model info ---------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        ctx = _active_ctx()
        if ctx is not None:
            return ReadyResponse(
                ready=True, model_version=ctx.model_id, detail="active model loaded"
            )
        return ReadyResponse(ready=False, model_version=None, detail="no active model")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        ctx = _active_ctx()
        if ctx is None:
            return ModelInfoResponse(
                model_version=None, model_name="none", features=[], loaded=False
            )
        meta = ctx.adapter.metadata()
        return ModelInfoResponse(
            model_version=meta.model_version,
            model_name=meta.name or "model",
            features=list(ctx.feature_names),
            loaded=True,
        )

    # ---- prediction ------------------------------------------------------------------
    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        ctx = _active_ctx()
        if ctx is None:
            raise HTTPException(status_code=503, detail="no active model")
        try:
            return predict_one(ctx, req, sink)
        except Exception as exc:  # noqa: BLE001 - surface adapter errors cleanly
            from drifttrace.adapters.base import AdapterError

            if isinstance(exc, AdapterError):
                raise HTTPException(status_code=422, detail=exc.message) from exc
            raise HTTPException(status_code=422, detail=f"prediction failed: {exc}") from exc

    # ---- monitoring / RCA ------------------------------------------------------------
    @app.get("/rca/latest")
    def rca_latest() -> dict:
        """Return the latest persisted monitoring/RCA report, or a not-available status."""
        import json

        latest = settings.reports_dir_path / "latest_rca.json"
        if not latest.exists():
            return {"available": False, "detail": "no monitoring report yet"}
        report = json.loads(latest.read_text(encoding="utf-8"))
        ctx = _active_ctx()
        graph_available = ctx is not None and ctx.graph is not None
        return {"available": True, "report": report, "dependencies_available": graph_available}

    @app.post("/demo/run-drift-test")
    def run_drift_test_endpoint(req: DriftTestRequest) -> dict:
        """Run a generic drift test against the active model's reference data.

        Perturbs a sample drawn from the reference distribution and runs it through the
        SAME drift -> RCA -> report pipeline used for live monitoring. No domain-specific
        transforms, no fabricated results.
        """
        ctx = _active_ctx()
        if ctx is None:
            raise HTTPException(status_code=503, detail="no active model")
        return run_drift_test(
            settings, ctx, intensity=req.intensity, feature=req.feature, n=req.n, seed=req.seed
        )

    # ---- explainability --------------------------------------------------------------
    @app.post("/explain")
    def explain(req: ExplainRequest) -> dict:
        """SHAP (primary) or LIME (secondary) explanation, off the hot path."""
        ctx = _active_ctx()
        if ctx is None:
            raise HTTPException(status_code=503, detail="no active model")

        from drifttrace.explain.explainer import (
            interpret_explanation,
            lime_explain_local,
            shap_explain_local,
        )

        features = dict(req.features)
        background = ctx.reference_frame
        raw_model = ctx.adapter.raw_model  # type: ignore[attr-defined]
        try:
            if req.method == "lime":
                exp = lime_explain_local(
                    raw_model,
                    features,  # type: ignore[arg-type]
                    ctx.feature_names,
                    background,
                    model_version=ctx.model_id,
                )
            else:
                exp = shap_explain_local(
                    raw_model,
                    features,  # type: ignore[arg-type]
                    ctx.feature_names,
                    model_version=ctx.model_id,
                    background=background,
                )
        except Exception as exc:  # noqa: BLE001 - explainability must not 500 the API
            raise HTTPException(status_code=500, detail=f"explanation failed: {exc}") from exc

        # Deterministic natural-language interpretation on top of SHAP/LIME (no LLM).
        try:
            task = ctx.adapter.metadata().task or "classification"
            result = ctx.adapter.predict_one(features)
            label = _prediction_label(raw_model, result, task)
            exp.interpretation = interpret_explanation(exp, prediction_label=label, task=task)
        except Exception as exc:  # noqa: BLE001 - interpretation must not 500 the API
            logger.warning("explanation interpretation failed: %s", exc)
        return exp.to_dict()

    @app.get("/metrics")
    def metrics() -> Any:
        """Prometheus-compatible metrics text (no Prometheus/Grafana installed)."""
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(METRICS.to_prometheus())

    # ---- model onboarding + activation -----------------------------------------------
    @app.post("/models/upload", response_model=UploadModelResponse)
    async def upload_model(file: UploadFile = File(...)) -> UploadModelResponse:  # noqa: B008
        """Upload a model bundle (.zip of model.pkl + reference.csv + optional graph.json).

        Also accepts a bare model file for convenience, but reference data is required to
        monitor, so a full bundle is recommended.
        """
        import tempfile
        from pathlib import Path as _Path

        from drifttrace.bundle.loader import BundleError
        from drifttrace.bundle.reference import ReferenceError

        suffix = _Path(file.filename or "bundle.zip").suffix or ".zip"
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = _Path(tmp.name)
        try:
            onboarded = REGISTRY.onboard_bundle(tmp_path, original_name=file.filename)
        except (BundleError, ReferenceError) as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"could not onboard bundle: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        return _upload_response(onboarded)

    def _active_model_response() -> ActiveModelResponse:
        ctx = _active_ctx()
        if ctx is None:
            return ActiveModelResponse(active=False, loaded=False)
        meta = ctx.adapter.metadata()
        return ActiveModelResponse(
            active=True,
            model_id=ctx.model_id,
            name=meta.name,
            framework=meta.framework,
            task=meta.task,
            model_version=meta.model_version,
            features=list(ctx.feature_names),
            dependencies_available=ctx.graph is not None,
            supports_proba=meta.supports_proba,
            loaded=True,
        )

    @app.get("/models/active", response_model=ActiveModelResponse)
    def active_model() -> ActiveModelResponse:
        """Return the model currently serving predictions, or an explicit no-model state."""
        return _active_model_response()

    @app.post("/models/{model_id}/activate", response_model=ActiveModelResponse)
    def activate_model(model_id: str) -> ActiveModelResponse:
        """Activate a supported onboarded model ("use this model")."""
        try:
            REGISTRY.activate(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            REGISTRY.deactivate()
            raise HTTPException(status_code=422, detail=f"could not activate model: {exc}") from exc
        # A new active model resets monitoring: clear any stale report from a prior model.
        _clear_latest_report(settings)
        logger.info("activated model %s", model_id)
        return _active_model_response()

    @app.post("/models/deactivate", response_model=ActiveModelResponse)
    def deactivate_model() -> ActiveModelResponse:
        """Clear the active model. There is no fallback; monitoring stops."""
        REGISTRY.deactivate()
        _clear_latest_report(settings)
        return _active_model_response()

    @app.get("/models/{model_id}", response_model=ModelStatusResponse)
    def model_status(model_id: str) -> ModelStatusResponse:
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
            active=(REGISTRY.active_model_id == onboarded.model_id),
            message=onboarded.message,
        )

    return app


def _clear_latest_report(settings: ServingSettings) -> None:
    """Remove the persisted latest RCA report so a new active model starts clean."""
    import contextlib

    latest = settings.reports_dir_path / "latest_rca.json"
    with contextlib.suppress(OSError):
        latest.unlink(missing_ok=True)


def run_drift_test(
    settings: ServingSettings,
    ctx: ActiveContext,
    *,
    intensity: float = 1.0,
    feature: str | None = None,
    n: int = 300,
    seed: int = 7,
) -> dict:
    """Perturb the active model's reference data and run the real monitoring pipeline.

    Draws ``n`` rows from the reference, applies a generic shift to numeric feature(s),
    predicts through the active model, builds prediction events, and runs the Phase 4
    windower + processor (drift -> RCA -> report -> alert). Writes latest_rca.json.
    """
    import numpy as np

    from drifttrace.alerting.alerter import Alerter
    from drifttrace.drift.config import load_drift_config
    from drifttrace.streaming.event import PredictionEvent, new_event_id
    from drifttrace.streaming.processing import run_scenario_over_events

    rng = np.random.default_rng(seed)
    ref = ctx.reference_frame
    feature_names = ctx.feature_names

    # Sample rows (with replacement) from the reference.
    idx = rng.integers(0, len(ref), size=n)
    sample = ref.iloc[idx].reset_index(drop=True)

    # Determine numeric features to perturb.
    numeric = [
        f for f in feature_names if f in sample.columns and pd.api.types.is_numeric_dtype(sample[f])
    ]
    targets = [feature] if feature and feature in numeric else numeric
    if intensity > 0 and targets:
        for f in targets:
            col = sample[f].to_numpy(dtype=float)
            std = float(np.nanstd(col)) or 1.0
            # A deterministic mean shift proportional to the feature's own spread.
            sample[f] = col + intensity * 1.5 * std

    # Predict through the active model and build standardized events.
    events: list[PredictionEvent] = []
    meta = ctx.adapter.metadata()
    for _, row in sample.iterrows():
        feats = {f: row[f] for f in feature_names if f in sample.columns}
        try:
            result = ctx.adapter.predict_one(feats)
        except Exception:  # noqa: BLE001 - skip rows the model rejects
            continue
        events.append(
            to_standard_event(result, meta, request_id=None, source="drift-test").model_copy(
                update={"event_id": new_event_id()}
            )
        )

    config = load_drift_config()
    graph = ctx.graph if ctx.graph is not None else _trivial_graph(ctx.feature_names)
    alerter = Alerter(
        webhook=None,
        cooldown_seconds=0,
        alert_log_path=settings.reports_dir_path / "alerts.jsonl",
    )
    outcomes = run_scenario_over_events(
        events,
        ctx.baseline,
        graph,
        config,
        alerter=alerter if ctx.graph is not None else None,
        reports_dir=settings.reports_dir_path,
        window_size=n,
        min_window_samples=min(30, max(10, n // 10)),
    )
    outcome = outcomes[-1] if outcomes else None
    return {
        "intensity": intensity,
        "feature": feature,
        "windows": len(outcomes),
        "dependencies_available": ctx.graph is not None,
        "outcome": outcome.to_dict() if outcome is not None else None,
    }


def _trivial_graph(feature_names: list[str]) -> Any:
    """A graph with no edges (each feature independent) for drift-only monitoring.

    Used when no dependency graph was provided: drift is still detected per feature, but
    no upstream root-cause tracing is possible (every drifted node is independent).
    """
    from drifttrace.bundle.reference import OUTPUT_NODE
    from drifttrace.graph.dag import DependencyGraph, NodeSpec

    specs = [NodeSpec(name=f, kind="raw_input") for f in feature_names]
    # A model_output node fed by all features keeps the graph connected + valid.
    specs.append(NodeSpec(name=OUTPUT_NODE, kind="model_output", parents=tuple(feature_names)))
    return DependencyGraph(specs)
