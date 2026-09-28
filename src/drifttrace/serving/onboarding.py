"""Model onboarding + activation for the local MVP (Phase 5).

Implements the "upload one model, DriftTrace figures out the rest" flow:

1. The user uploads a single supported model file.
2. DriftTrace inspects it automatically (framework, task, features, proba support).
3. Reference data and dependency information are reused from the existing project
   context when available; only genuinely missing information is requested.
4. A supported model can be *activated* ("Use this model") so ``/predict`` serves it.

This is an in-memory onboarding registry for the local MVP - it records what was
detected so the UI can show a compact "MODEL READY" confirmation, and it keeps the
uploaded file on disk (under the model store) so an onboarded model can become the
active model. It does NOT replace the trained loan model, which remains the default
and fallback when nothing custom is activated. There is no database-backed registry.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from drifttrace.adapters.base import ModelAdapter
from drifttrace.adapters.registry import build_adapter, inspect_model_file
from drifttrace.config import get_paths


def _reference_available() -> bool:
    """Reference/baseline distribution exists if a drift baseline is present."""
    return (get_paths().artifacts / "baseline.json").exists()


def _graph_available() -> bool:
    """A declared dependency graph exists (enables root-cause tracing)."""
    return get_paths().graph_yaml.exists()


@dataclass
class OnboardedModel:
    """A model DriftTrace has inspected during onboarding (a *registered* model)."""

    model_id: str
    supported: bool
    framework: str | None
    name: str | None
    task: str | None
    n_features: int | None
    supports_proba: bool
    features: list[str] = field(default_factory=list)
    reference_available: bool = False
    dependencies_available: bool = False
    missing: list[str] = field(default_factory=list)
    message: str | None = None
    # Local path where the uploaded file is retained so it can be activated. None for
    # unsupported uploads (which are not retained).
    stored_path: Path | None = None

    @property
    def ready_to_monitor(self) -> bool:
        # Ready when supported and a reference distribution exists. Dependency info is
        # not strictly required to monitor, but its absence limits root-cause tracing.
        return self.supported and self.reference_available


class OnboardingRegistry:
    """In-memory registry of onboarded (registered) models, per serving process.

    Distinguishes two states:
    - *registered*: a model that has been uploaded + inspected (kept in ``_models``).
    - *active*: the single registered model currently selected to serve predictions
      (``active_model_id``). When ``None``, the default loan model serves.
    """

    def __init__(self) -> None:
        self._models: dict[str, OnboardedModel] = {}
        self.active_model_id: str | None = None

    # ---- registration (upload + inspect) ---------------------------------------------
    def onboard(self, model_path: Path, *, original_name: str | None = None) -> OnboardedModel:
        """Inspect an uploaded model file and record what was determined.

        Supported models are retained under the model store so they can be activated;
        unsupported uploads are not retained.
        """
        ref = _reference_available()
        graph = _graph_available()
        result = inspect_model_file(model_path, reference_available=ref, graph_available=graph)
        model_id = "mdl-" + uuid.uuid4().hex[:12]

        stored_path: Path | None = None
        if result.supported:
            store = get_paths().model_store
            store.mkdir(parents=True, exist_ok=True)
            suffix = model_path.suffix or ".pkl"
            stored_path = store / f"{model_id}{suffix}"
            shutil.copyfile(model_path, stored_path)

        onboarded = OnboardedModel(
            model_id=model_id,
            supported=result.supported,
            framework=result.framework,
            name=result.name or original_name,
            task=result.task,
            n_features=result.n_features,
            supports_proba=result.supports_proba,
            features=result.features,
            reference_available=ref,
            dependencies_available=graph,
            missing=result.missing,
            message=result.message,
            stored_path=stored_path,
        )
        if onboarded.supported:
            self._models[model_id] = onboarded
        return onboarded

    def get(self, model_id: str) -> OnboardedModel | None:
        return self._models.get(model_id)

    # ---- activation ------------------------------------------------------------------
    def activate(self, model_id: str) -> OnboardedModel:
        """Mark a registered, supported model as the active model.

        Raises ``KeyError`` if unknown and ``ValueError`` if it cannot be activated.
        """
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(model_id)
        if not model.supported or model.stored_path is None:
            raise ValueError("model is not supported and cannot be activated")
        if not model.ready_to_monitor:
            raise ValueError("reference data is required before a model can be activated")
        self.active_model_id = model_id
        return model

    def deactivate(self) -> None:
        """Clear the active model, restoring the default loan model."""
        self.active_model_id = None

    def active(self) -> OnboardedModel | None:
        if self.active_model_id is None:
            return None
        return self._models.get(self.active_model_id)

    def build_active_adapter(self) -> ModelAdapter | None:
        """Load + wrap the active model in an adapter, or return None if none active.

        The active model file is loaded from the model store and wrapped through the
        same adapter boundary as the default model. Transform params are the shared
        deterministic loan-feature params (the MVP feature chain), consistent with how
        the model was inspected on upload.
        """
        model = self.active()
        if model is None or model.stored_path is None:
            return None
        from drifttrace.adapters.registry import _default_params, _load_object

        raw = _load_object(model.stored_path)
        return build_adapter(
            raw,
            _default_params(),
            model_id=model.model_id,
            model_version="active",
        )


# Process-wide onboarding registry.
REGISTRY = OnboardingRegistry()
