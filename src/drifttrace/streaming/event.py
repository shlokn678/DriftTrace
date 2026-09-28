"""Prediction event schema shared across serving, streaming, and the monitor (FR-7.4, FR-13).

A :class:`PredictionEvent` is the single message contract that the FastAPI service
emits after every successful prediction and that the monitor consumes from the event
source (file replay or Redpanda). It carries the fields the design requires: model
version, timestamp, input features, the prediction/result, and identifiers (FR-7.4).

This module is pure (pydantic + stdlib only): no broker, FastAPI, or Docker import, so
it is unit-testable in isolation (NFR-8) and safe to import from any adapter.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_event_id() -> str:
    """Generate a unique event identifier."""
    return uuid.uuid4().hex


class PredictionEvent(BaseModel):
    """A single prediction event.

    Fields:
        schema_version: contract version, for forward compatibility.
        event_id: unique id for this event (dedup / tracing).
        request_id: id of the originating prediction request (may repeat for batch).
        model_id: model-agnostic identifier of the producing model (Phase 5). Optional
            and backward compatible; older events omit it.
        model_version: the registered model version that produced the prediction.
        timestamp: ISO-8601 UTC time the event was created.
        features: the model-input feature values (feature name -> value).
        prediction: the discrete class label for classifiers (None for regressors).
        probability: the positive/decision score, when the model supports it.
        output: the model's numeric output used for output-drift monitoring - the
            predicted class for classifiers or the predicted value for regressors.
        group: optional sensitive/group attribute value for fairness monitoring.
        source: free-form origin marker (e.g. "api", "replay", "drift-test").
    """

    schema_version: str = SCHEMA_VERSION
    event_id: str = Field(default_factory=new_event_id)
    request_id: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    timestamp: str = Field(default_factory=_utc_now_iso)
    # Feature values may be numeric (continuous) or string labels (categorical).
    features: dict[str, float | str] = Field(default_factory=dict)
    prediction: int | None = None
    probability: float | None = None
    output: float | None = None
    group: str | None = None
    source: str = "api"

    def to_json(self) -> str:
        """Serialize to a compact JSON string (used as the broker message value)."""
        return self.model_dump_json()

    def to_bytes(self) -> bytes:
        """Serialize to UTF-8 JSON bytes for the broker."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, raw: str | bytes) -> PredictionEvent:
        """Parse an event from a JSON string or bytes (broker/file message value)."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.model_validate(json.loads(raw))
