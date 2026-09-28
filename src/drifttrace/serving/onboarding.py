"""Minimal model onboarding (Phase 5).

Implements the "upload one model, DriftTrace figures out the rest" flow:

1. The user uploads a single supported model file.
2. DriftTrace inspects it automatically (framework, task, features, proba support).
3. Reference data and dependency information are reused from the existing project
   context when available; only genuinely missing information is requested.

This is an in-memory onboarding registry for the local MVP - it records what was
detected so the UI can show a compact "MODEL READY" confirmation. It does not replace
the trained loan model that already serves predictions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from drifttrace.adapters.registry import inspect_model_file
from drifttrace.config import get_paths


def _reference_available() -> bool:
    """Reference/baseline distribution exists if a drift baseline is present."""
    return (get_paths().artifacts / "baseline.json").exists()


def _graph_available() -> bool:
    """A declared dependency graph exists (enables root-cause tracing)."""
    return get_paths().graph_yaml.exists()


@dataclass
class OnboardedModel:
    """A model DriftTrace has inspected during onboarding."""

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

    @property
    def ready_to_monitor(self) -> bool:
        # Ready when supported and a reference distribution exists. Dependency info is
        # not strictly required to monitor, but its absence limits root-cause tracing.
        return self.supported and self.reference_available


class OnboardingRegistry:
    """In-memory registry of onboarded models (per serving process)."""

    def __init__(self) -> None:
        self._models: dict[str, OnboardedModel] = {}

    def onboard(self, model_path: Path, *, original_name: str | None = None) -> OnboardedModel:
        """Inspect an uploaded model file and record what was determined."""
        ref = _reference_available()
        graph = _graph_available()
        result = inspect_model_file(model_path, reference_available=ref, graph_available=graph)
        model_id = "mdl-" + uuid.uuid4().hex[:12]
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
        )
        if onboarded.supported:
            self._models[model_id] = onboarded
        return onboarded

    def get(self, model_id: str) -> OnboardedModel | None:
        return self._models.get(model_id)


# Process-wide onboarding registry.
REGISTRY = OnboardingRegistry()
