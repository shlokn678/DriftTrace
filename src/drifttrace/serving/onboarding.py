"""Model bundle onboarding + active-model management (model-agnostic).

A user uploads a bundle (``model.pkl`` + ``reference.csv`` + optional ``graph.json``).
DriftTrace inspects it, and the user activates it. The active model then:
  - serves ``/predict`` (generic feature-vector payload),
  - defines the drift baseline (from its reference data),
  - defines the optional dependency graph (from its graph.json),
  - is monitored via incoming prediction events and the drift-test.

There is NO built-in / default / fallback model. A fresh install starts with no model
registered and no active model. The registry is in-memory for the local MVP - no
database.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from drifttrace.adapters.base import ModelAdapter
from drifttrace.adapters.registry import build_adapter, inspect_model, load_model_object
from drifttrace.bundle.graph_json import GraphParseError, load_graph_json
from drifttrace.bundle.loader import extract_bundle_zip, load_bundle_dir
from drifttrace.bundle.reference import (
    OUTPUT_NODE,
    ReferenceError,
    baseline_from_profile,
    profile_reference,
)
from drifttrace.config import get_paths
from drifttrace.drift.baseline import Baseline
from drifttrace.graph.dag import DependencyGraph


@dataclass
class OnboardedModel:
    """A model + reference (+ optional graph) DriftTrace has inspected (registered)."""

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
    reference_rows: int | None = None
    # Retained bundle paths so the model can be activated.
    bundle_dir: Path | None = None
    model_path: Path | None = None
    reference_path: Path | None = None
    graph_path: Path | None = None

    @property
    def ready_to_monitor(self) -> bool:
        # A supported model with reference data is monitorable. A graph only adds RCA.
        return self.supported and self.reference_available


@dataclass
class ActiveContext:
    """Everything needed to serve + monitor the active model."""

    model_id: str
    adapter: ModelAdapter
    baseline: Baseline
    graph: DependencyGraph | None
    feature_names: list[str]
    reference_frame: pd.DataFrame


class OnboardingRegistry:
    """In-memory registry of onboarded models + the single active model (per process)."""

    def __init__(self) -> None:
        self._models: dict[str, OnboardedModel] = {}
        self.active_model_id: str | None = None
        self._active_ctx: ActiveContext | None = None

    # ---- registration ----------------------------------------------------------------
    def onboard_bundle(self, source: Path, *, original_name: str | None = None) -> OnboardedModel:
        """Inspect an uploaded bundle (a .zip or a directory of the three files).

        Raises :class:`BundleError` / :class:`ReferenceError` / :class:`GraphParseError`
        with user-facing messages when the bundle is unusable.
        """
        model_id = "mdl-" + uuid.uuid4().hex[:12]
        store = get_paths().model_store / model_id
        store.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            loaded = load_bundle_dir(source)
            # Copy into the model store so activation survives temp cleanup.
            import shutil

            model_dst = store / loaded.model_path.name
            ref_dst = store / "reference.csv"
            shutil.copyfile(loaded.model_path, model_dst)
            shutil.copyfile(loaded.reference_path, ref_dst)
            graph_dst: Path | None = None
            if loaded.graph_path is not None:
                graph_dst = store / "graph.json"
                shutil.copyfile(loaded.graph_path, graph_dst)
            loaded_model, loaded_ref, loaded_graph = model_dst, ref_dst, graph_dst
        else:
            loaded = extract_bundle_zip(source, store)
            loaded_model, loaded_ref, loaded_graph = (
                loaded.model_path,
                loaded.reference_path,
                loaded.graph_path,
            )

        # Load + inspect the model.
        model = load_model_object(loaded_model)
        reference_frame = _read_reference(loaded_ref)
        profile = profile_reference(reference_frame)

        graph_available = loaded_graph is not None
        inspection = inspect_model(
            model,
            reference_features=profile.feature_names,
            graph_available=graph_available,
        )

        # If a graph was provided, validate it now so problems surface at onboarding.
        graph_message: str | None = None
        if graph_available and loaded_graph is not None:
            try:
                load_graph_json(loaded_graph, feature_names=profile.feature_names)
            except GraphParseError as exc:
                graph_available = False
                graph_message = exc.message

        onboarded = OnboardedModel(
            model_id=model_id,
            supported=inspection.supported,
            framework=inspection.framework,
            name=inspection.name,
            task=inspection.task,
            n_features=inspection.n_features or profile.n_features,
            supports_proba=inspection.supports_proba,
            features=inspection.features or profile.feature_names,
            reference_available=True,
            dependencies_available=graph_available,
            missing=[] if graph_available else ["dependencies"],
            message=graph_message or inspection.message,
            reference_rows=profile.n_rows,
            bundle_dir=store,
            model_path=loaded_model,
            reference_path=loaded_ref,
            graph_path=loaded_graph if graph_available else None,
        )
        if onboarded.supported:
            self._models[model_id] = onboarded
        return onboarded

    def get(self, model_id: str) -> OnboardedModel | None:
        return self._models.get(model_id)

    # ---- activation -------------------------------------------------------------------
    def activate(self, model_id: str) -> ActiveContext:
        """Activate a registered, supported model. Builds its adapter, baseline, graph."""
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(model_id)
        if not model.supported or model.model_path is None or model.reference_path is None:
            raise ValueError("model is not supported and cannot be activated")

        raw = load_model_object(model.model_path)
        reference_frame = _read_reference(model.reference_path)
        profile = profile_reference(reference_frame)

        adapter = build_adapter(
            raw,
            profile.feature_names,
            model_id=model_id,
            model_version="active",
            name=model.name,
            task=model.task,
        )

        # Baseline: reference feature distributions + the model's reference output
        # distribution (so output drift can be detected).
        output_values = _reference_outputs(adapter, reference_frame, profile.feature_names)
        baseline = baseline_from_profile(
            profile, model_version=model_id, output_values=output_values
        )

        graph: DependencyGraph | None = None
        if model.graph_path is not None:
            graph = load_graph_json(model.graph_path, feature_names=profile.feature_names)

        ctx = ActiveContext(
            model_id=model_id,
            adapter=adapter,
            baseline=baseline,
            graph=graph,
            feature_names=profile.feature_names,
            reference_frame=reference_frame,
        )
        self.active_model_id = model_id
        self._active_ctx = ctx
        return ctx

    def deactivate(self) -> None:
        """Clear the active model. There is no fallback - monitoring stops."""
        self.active_model_id = None
        self._active_ctx = None

    def active(self) -> OnboardedModel | None:
        if self.active_model_id is None:
            return None
        return self._models.get(self.active_model_id)

    def active_context(self) -> ActiveContext | None:
        return self._active_ctx


def _read_reference(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - clean message for the UI
        raise ReferenceError(
            "The reference data could not be read as CSV.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    if frame.shape[1] == 0 or len(frame) == 0:
        raise ReferenceError("The reference data is empty.")
    return frame


def _reference_outputs(
    adapter: ModelAdapter,
    reference_frame: pd.DataFrame,
    feature_names: list[str],
) -> list[float] | None:
    """Run the model over a bounded reference sample to capture its output baseline."""
    cols = [c for c in feature_names if c in reference_frame.columns]
    if not cols:
        return None
    sample = reference_frame[cols].head(1000)
    outputs: list[float] = []
    for _, row in sample.iterrows():
        try:
            res = adapter.predict_one({c: row[c] for c in cols})
        except Exception:  # noqa: BLE001 - skip rows the model rejects
            continue
        if res.output is not None:
            outputs.append(float(res.output))
    return outputs or None


# The canonical output node name, re-exported for callers building windows/graphs.
__all__ = ["OnboardingRegistry", "OnboardedModel", "ActiveContext", "REGISTRY", "OUTPUT_NODE"]

# Process-wide onboarding registry.
REGISTRY = OnboardingRegistry()
