"""Model adapter layer (Phase 5).

Separates model-specific prediction logic from the model-agnostic DriftTrace core.
The core (drift, RCA, alerting, reports) consumes standardized prediction events and
never imports a specific ML framework. The first supported adapter is scikit-learn.

Public surface:
    ModelAdapter        - the small adapter interface (ABC)
    ModelMetadata       - detected model metadata
    FeatureSchema       - detected input feature schema
    AdapterError        - raised for unsupported/incompatible models
    to_standard_event   - convert an adapter prediction to a StandardizedPredictionEvent
"""

from drifttrace.adapters.base import (
    AdapterError,
    FeatureSchema,
    ModelAdapter,
    ModelMetadata,
    PredictionResult,
    to_standard_event,
)

__all__ = [
    "AdapterError",
    "FeatureSchema",
    "ModelAdapter",
    "ModelMetadata",
    "PredictionResult",
    "to_standard_event",
]
