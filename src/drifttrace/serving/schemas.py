"""Request/response schemas for the serving API (model-agnostic)."""

from __future__ import annotations

from pydantic import BaseModel, Field

# A single feature value may be numeric (continuous) or a string label (categorical).
FeatureValue = float | int | str


class PredictRequest(BaseModel):
    """A single prediction request.

    ``features`` maps the active model's feature names to values. The set of features is
    defined by the active model (from its reference data), not by DriftTrace.
    """

    features: dict[str, FeatureValue] = Field(..., description="feature name -> value")
    request_id: str | None = None


class ExplainRequest(BaseModel):
    """An explanation request (off the prediction hot path)."""

    features: dict[str, FeatureValue] = Field(..., description="feature name -> value")
    method: str = Field("shap", description="'shap' (primary) or 'lime' (secondary)")


class DriftTestRequest(BaseModel):
    """Run a generic drift test: perturb the active model's reference data and evaluate.

    ``intensity`` scales the injected shift (0 = none, 1 = strong). ``feature`` optionally
    targets a single feature; when omitted, all numeric features are perturbed.
    """

    intensity: float = Field(1.0, ge=0.0, le=5.0)
    feature: str | None = None
    n: int = Field(300, ge=20, le=5000)
    seed: int = Field(7)


class PredictResponse(BaseModel):
    """A single prediction response."""

    request_id: str | None
    model_version: str | None
    prediction: int | None
    probability: float | None
    output: float | None
    features: dict[str, FeatureValue]
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


# ---- Model onboarding / activation schemas -------------------------------------------
class UploadModelResponse(BaseModel):
    """Result of inspecting an uploaded model bundle.

    ``missing`` lists only genuinely undeterminable information (e.g. ``dependencies``
    when no graph.json was provided). ``message`` is a user-friendly note.
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
    reference_rows: int | None = None
    missing: list[str] = Field(default_factory=list)
    ready_to_monitor: bool = False
    message: str | None = None


class ModelStatusResponse(BaseModel):
    """Registration/onboarding status for a model."""

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
    """The model currently serving predictions, or an explicit no-model state."""

    active: bool
    model_id: str | None = None
    name: str | None = None
    framework: str | None = None
    task: str | None = None
    model_version: str | None = None
    features: list[str] = Field(default_factory=list)
    dependencies_available: bool = False
    supports_proba: bool = False
    loaded: bool = False
