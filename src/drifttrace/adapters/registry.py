"""Adapter registry + automatic model inspection (Phase 5).

Given a loaded model object (or a saved model file), determine which adapter family
supports it and build the adapter. Only scikit-learn is supported in this phase;
unsupported models raise a clear :class:`AdapterError`.

DriftTrace never fabricates metadata. When something cannot be determined it stays
unknown and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drifttrace.adapters.base import AdapterError, ModelAdapter
from drifttrace.adapters.sklearn_adapter import SklearnAdapter
from drifttrace.features.transform import TransformParams, fit_params

# File extensions we will attempt to load. Others are rejected up front with a clear
# message rather than a stack trace.
SUPPORTED_SUFFIXES = {".pkl", ".pickle", ".joblib"}


@dataclass
class InspectionResult:
    """What auto-inspection could determine about an uploaded model."""

    supported: bool
    framework: str | None
    name: str | None
    task: str | None
    n_features: int | None
    supports_proba: bool
    features: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # info that must be supplied
    message: str | None = None  # user-facing note when unsupported / limited

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
    # Duck-typing fallback: sklearn estimators expose get_params + predict.
    return hasattr(model, "get_params") and (
        hasattr(model, "predict") or hasattr(model, "predict_proba")
    )


def build_adapter(
    model: Any,
    transform_params: TransformParams,
    *,
    model_id: str = "drifttrace-loan-default",
    model_version: str | None = None,
) -> ModelAdapter:
    """Build the appropriate adapter for a loaded model object.

    Raises :class:`AdapterError` if no supported adapter matches (unsupported model).
    """
    if _looks_like_sklearn(model):
        adapter = SklearnAdapter(
            model,
            transform_params,
            model_id=model_id,
            model_version=model_version,
        )
        adapter.validate()
        return adapter
    raise AdapterError(
        "Unsupported model format.",
        detail=(
            f"no adapter for model type {type(model).__module__}.{type(model).__name__}; "
            "scikit-learn is the only supported adapter in this phase"
        ),
    )


def _load_object(path: Path) -> Any:
    """Load a saved model object from a supported file. Raises AdapterError on failure."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AdapterError(
            "Unsupported model format.",
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


def inspect_model_file(
    path: Path,
    *,
    reference_available: bool,
    graph_available: bool,
) -> InspectionResult:
    """Auto-inspect a saved model file and report what DriftTrace could determine.

    Only asks (via ``missing``) for what genuinely cannot be determined:
    - reference data (needed for drift detection) when not already available;
    - feature dependencies (needed for root-cause tracing) when not available.
    """
    try:
        model = _load_object(path)
    except AdapterError as exc:
        return InspectionResult(
            supported=False,
            framework=None,
            name=None,
            task=None,
            n_features=None,
            supports_proba=False,
            message=exc.message,
        )

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

    # Build a provisional adapter to read metadata + schema. Transform params are not
    # stored inside a bare model file, so a neutral default is used purely for
    # introspection here; the real serving path uses the trained params.
    adapter = SklearnAdapter(model, _default_params())
    meta = adapter.metadata()
    schema = adapter.feature_schema()

    missing: list[str] = []
    if not reference_available:
        missing.append("reference_data")
    if not graph_available:
        missing.append("dependencies")

    message: str | None = None
    if "dependencies" in missing:
        message = "Root-cause tracing is limited without feature dependency information."

    return InspectionResult(
        supported=True,
        framework=meta.framework,
        name=meta.name,
        task=meta.task,
        n_features=schema.n_features,
        supports_proba=meta.supports_proba,
        features=list(schema.features),
        missing=missing,
        message=message,
    )


def _default_params() -> TransformParams:
    """Neutral transform params for introspection only (not used for real serving)."""
    import numpy as np

    return fit_params(np.array([1000.0, 2000.0, 3000.0], dtype=float))
