"""Model adapter interface and standardized-event conversion (Phase 5).

The adapter interface is intentionally small. It exposes only what the DriftTrace core
and the serving layer need, and it never leaks framework-specific objects (e.g. an
sklearn ``Pipeline``) into the generic core. A concrete adapter (e.g. the sklearn
adapter) implements these methods for one model family.

Pure module: numpy + stdlib only, so it is importable without any ML framework (NFR-8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np

from drifttrace.streaming.event import PredictionEvent, new_event_id


class AdapterError(Exception):
    """Raised for unsupported model formats or incompatible inputs.

    Carries a user-friendly ``message`` (safe to show in a UI) and an optional
    ``detail`` (developer context, e.g. an underlying exception string).
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass
class ModelMetadata:
    """Metadata DriftTrace could detect about a model.

    Any field may be ``None`` when it cannot be reliably determined. DriftTrace never
    fabricates metadata; unknown values stay ``None``.
    """

    framework: str  # e.g. "scikit-learn"
    name: str | None = None
    task: str | None = None  # e.g. "binary_classification"
    model_version: str | None = None
    model_id: str | None = None
    supports_proba: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureSchema:
    """The model's input feature schema, where available.

    ``features`` is the ordered list of feature names the model consumes. ``raw_inputs``
    are the user-supplied inputs (a subset of features that are not derived). Both may
    be empty when the schema cannot be determined.
    """

    features: list[str] = field(default_factory=list)
    raw_inputs: list[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        return asdict(self) | {"n_features": self.n_features}


@dataclass
class PredictionResult:
    """The model-agnostic result of a single prediction.

    ``features`` is the feature vector the model consumed, so the monitoring core sees
    the same values the model saw. ``prediction`` is the discrete class for classifiers
    (``None`` for regressors); ``probability`` is the positive/decision score when
    available; ``output`` is the numeric value used for output-drift monitoring
    (predicted class for classifiers, predicted value for regressors).
    """

    prediction: int | None
    probability: float | None
    features: dict[str, float | str]
    output: float | None = None


class ModelAdapter(ABC):
    """Small, framework-agnostic interface DriftTrace uses to talk to a model.

    Concrete adapters wrap one model family. The core never checks the concrete type.
    """

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return detected model metadata (unknown fields stay None)."""

    @abstractmethod
    def feature_schema(self) -> FeatureSchema:
        """Return the input feature schema, where available."""

    @abstractmethod
    def supports_proba(self) -> bool:
        """Whether the model can produce class probabilities."""

    @abstractmethod
    def predict_one(self, features: dict[str, float | str]) -> PredictionResult:
        """Predict for a single record given its feature values (name -> value).

        Implementations order the values by the model's expected feature names and
        return the feature vector used, so the caller stays model-agnostic.
        """

    def validate(self) -> None:
        """Raise :class:`AdapterError` if the wrapped model is not usable.

        Default implementation performs a light check via metadata/schema; concrete
        adapters may override with a stronger check.
        """
        meta = self.metadata()
        if not meta.framework:
            raise AdapterError("Unsupported model: framework could not be determined.")


def to_standard_event(
    result: PredictionResult,
    metadata: ModelMetadata,
    *,
    request_id: str | None = None,
    group: str | None = None,
    source: str = "api",
) -> PredictionEvent:
    """Convert an adapter :class:`PredictionResult` to a standardized PredictionEvent.

    This is the single conversion point between the model-specific adapter output and
    the model-agnostic event the drift/RCA/alerting core consumes.
    """
    features: dict[str, float | str] = {
        k: (v if isinstance(v, str) else float(v)) for k, v in result.features.items()
    }
    return PredictionEvent(
        event_id=new_event_id(),
        request_id=request_id,
        model_id=metadata.model_id,
        model_version=metadata.model_version,
        features=features,
        prediction=(None if result.prediction is None else int(result.prediction)),
        probability=(None if result.probability is None else float(result.probability)),
        output=(None if result.output is None else float(result.output)),
        group=group,
        source=source,
    )


def _as_float_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)
