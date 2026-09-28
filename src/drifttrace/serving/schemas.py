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


class RunScenarioRequest(BaseModel):
    """Request to run a deterministic drift-injection scenario (FR-18)."""

    scenario: str = Field(
        "control",
        description="control | income_annual | mid_chain | two_roots",
    )
    n: int = Field(300, ge=30, le=5000, description="events to generate")
    seed: int = Field(7, description="deterministic seed")


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


# ---- Phase 5: model onboarding schemas ------------------------------------------------
class UploadModelResponse(BaseModel):
    """Result of inspecting an uploaded model file (Phase 5).

    ``supported`` indicates whether DriftTrace can use the model. ``missing`` lists
    only the information that genuinely could not be determined and must be supplied
    (e.g. ``reference_data``, ``dependencies``). ``message`` is a user-friendly note.
    """

    model_id: str | None = None
    supported: bool
    framework: str | None = None
    name: str | None = None
    task: str | None = None
    n_features: int | None = None
    supports_proba: bool = False
    features: list[str] = Field(default_factory=list)
    reference_available: bool = False
    dependencies_available: bool = False
    missing: list[str] = Field(default_factory=list)
    ready_to_monitor: bool = False
    message: str | None = None


class ModelStatusResponse(BaseModel):
    """Registration/onboarding status for a model (Phase 5)."""

    model_id: str
    supported: bool
    framework: str | None = None
    name: str | None = None
    task: str | None = None
    n_features: int | None = None
    reference_available: bool = False
    dependencies_available: bool = False
    ready_to_monitor: bool = False
    active: bool = False
    message: str | None = None


class ActiveModelResponse(BaseModel):
    """The model currently serving predictions (Phase 5).

    ``is_custom`` distinguishes an onboarded/activated model from the default loan
    model that ships with the project and serves as the fallback.
    """

    is_custom: bool
    model_id: str | None = None
    name: str
    framework: str | None = None
    task: str | None = None
    model_version: str | None = None
    features: list[str] = Field(default_factory=list)
    loaded: bool = False
