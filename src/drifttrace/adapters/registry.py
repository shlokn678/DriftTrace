"""Adapter registry + model inspection (model-agnostic).

Detects the model family and builds the appropriate :class:`ModelAdapter`. The MVP
implements one adapter - scikit-learn. Future adapters (XGBoost, LightGBM, ONNX,
PyTorch) can be registered here without touching the monitoring/RCA core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drifttrace.adapters.base import AdapterError, ModelAdapter
from drifttrace.adapters.sklearn_adapter import (
    SklearnAdapter,
    detect_feature_names,
    detect_task,
)

SUPPORTED_SUFFIXES = {".pkl", ".pickle", ".joblib"}


@dataclass
class InspectionResult:
    """What auto-inspection could determine about an uploaded model + reference."""

    supported: bool
    framework: str | None
    name: str | None
    task: str | None
    n_features: int | None
    supports_proba: bool
    features: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "framework": self.framework,
            "name": self.name,
            "task": self.task,
            "n_features": self.n_features,
            "supports_proba": self.supports_proba,
            "features": self.features,
            "missing": self.missing,
            "message": self.message,
        }


def _looks_like_sklearn(model: Any) -> bool:
    """Detect an sklearn-style estimator without importing sklearn eagerly."""
    module = type(model).__module__ or ""
    if module.startswith("sklearn") or module.startswith("imblearn"):
        return True
    return hasattr(model, "get_params") and (
        hasattr(model, "predict") or hasattr(model, "predict_proba")
    )


def build_adapter(
    model: Any,
    feature_names: list[str],
    *,
    model_id: str,
    model_version: str | None = None,
    name: str | None = None,
    task: str | None = None,
) -> ModelAdapter:
    """Build the appropriate adapter for a loaded model over ``feature_names``.

    Raises :class:`AdapterError` if no supported adapter matches.
    """
    if _looks_like_sklearn(model):
        adapter = SklearnAdapter(
            model,
            feature_names,
            model_id=model_id,
            model_version=model_version,
            name=name,
            task=task,
        )
        adapter.validate()
        return adapter
    raise AdapterError(
        "Unsupported model format.",
        detail=(
            f"no adapter for model type {type(model).__module__}.{type(model).__name__}; "
            "scikit-learn is the only supported adapter"
        ),
    )


def load_model_object(path: Path) -> Any:
    """Load a saved model object from a supported file. Raises AdapterError on failure."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AdapterError(
            "Unsupported model file format.",
            detail=f"file extension '{suffix}' is not supported "
            f"(supported: {sorted(SUPPORTED_SUFFIXES)})",
        )
    try:
        if suffix == ".joblib":
            import joblib

            return joblib.load(path)
        import cloudpickle

        with path.open("rb") as fh:
            return cloudpickle.load(fh)
    except AdapterError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean message, keep detail
        raise AdapterError(
            "We could not read that model file.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def inspect_model(
    model: Any,
    *,
    reference_features: list[str] | None = None,
    graph_available: bool = False,
) -> InspectionResult:
    """Inspect a loaded model (with optional reference feature names) generically.

    ``missing`` reports only genuinely undeterminable information (a dependency graph,
    when absent). Reference data is validated separately by the onboarding layer.
    """
    if not _looks_like_sklearn(model):
        return InspectionResult(
            supported=False,
            framework=None,
            name=None,
            task=None,
            n_features=None,
            supports_proba=False,
            message="Unsupported model format. scikit-learn models are supported.",
        )

    features = detect_feature_names(model, fallback=reference_features)
    task = detect_task(model)
    supports_proba = hasattr(model, "predict_proba")

    missing: list[str] = []
    message: str | None = None
    if not graph_available:
        missing.append("dependencies")
        message = "Root-cause tracing is limited without a dependency graph."

    return InspectionResult(
        supported=True,
        framework="scikit-learn",
        name=type(model).__name__,
        task=task,
        n_features=len(features) if features else None,
        supports_proba=supports_proba,
        features=list(features),
        missing=missing,
        message=message,
    )
