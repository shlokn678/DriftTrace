"""Request/response schemas for the serving API (FR-7)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """A single prediction request.

    Only the raw input ``income`` is required; the shared feature transform derives
    ``credit_score`` and ``risk_score`` so training and serving never diverge (FR-3.3).
    An optional ``request_id`` supports tracing through the event pipeline.
    """

    income: float = Field(..., ge=0, description="Raw monthly income (>= 0)")
    request_id: str | None = None


class BatchPredictRequest(BaseModel):
    """A batch of prediction requests."""

    items: list[PredictRequest] = Field(..., min_length=1)


class ExplainRequest(BaseModel):
    """An explanation request (off the prediction hot path, FR-14.4)."""

    income: float = Field(..., ge=0, description="Raw monthly income (>= 0)")
    method: str = Field("shap", description="'shap' (primary) or 'lime' (secondary)")


class PredictResponse(BaseModel):
    """A single prediction response (FR-7 AC-3)."""

    request_id: str | None
    model_version: str | None
    prediction: int
    probability: float
    features: dict[str, float]
    event_emitted: bool


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str | None
    detail: str


class ModelInfoResponse(BaseModel):
    model_version: str | None
    model_name: str
    features: list[str]
    loaded: bool
