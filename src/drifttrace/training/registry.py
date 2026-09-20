"""MLflow tracking and model registry adapter (FR-5).

A thin adapter over MLflow: it logs parameters, metrics, and artifacts, records the
lifecycle evidence (dataset version, code commit, pipeline execution id), and
registers the model version. MLflow runs locally with a file/SQLite backend and a
local artifact root (decision D-6); no hosted server or account is assumed.

Kept importable without a running MLflow server: functions accept a tracking URI and
default to a local ``mlruns`` directory under the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from drifttrace.config import get_paths
from drifttrace.training.evidence import LifecycleEvidence
from drifttrace.training.train import TrainResult

REGISTERED_MODEL_NAME = "drifttrace-loan-default"
EXPERIMENT_NAME = "drifttrace"


def default_tracking_uri() -> str:
    """Local SQLite tracking URI under the repo (decision D-6).

    MLflow 3.x puts the plain filesystem store in maintenance mode and requires a
    database backend for the tracking + registry features used here, so the local
    backend is SQLite (``mlflow.db``). Artifacts still live on the local filesystem.
    No hosted server or account is involved.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    root = get_paths().root
    root.mkdir(parents=True, exist_ok=True)
    db_path = (root / "mlflow.db").as_posix()
    return f"sqlite:///{db_path}"


def _artifact_dir_for(uri: str, artifact_dir: Path | None) -> Path:
    """Choose a local filesystem artifact directory.

    Co-locate artifacts with the SQLite DB when possible so a tmp/test DB keeps its
    artifacts in the same temporary directory (no cwd pollution).
    """
    if artifact_dir is not None:
        return artifact_dir
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.removeprefix("sqlite:///"))
        return db_path.parent / "mlartifacts"
    return get_paths().root / "mlartifacts"


def _ensure_experiment(uri: str, artifact_dir: Path | None) -> None:
    """Set (creating if needed) the DriftTrace experiment with a local artifact root."""
    art = _artifact_dir_for(uri, artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=art.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


@dataclass
class LoggedRun:
    """Identifiers returned after logging a run."""

    run_id: str
    model_version: str | None
    tracking_uri: str


def log_training_run(
    estimator: Pipeline,
    result: TrainResult,
    evidence: LifecycleEvidence,
    *,
    tracking_uri: str | None = None,
    register: bool | None = None,
    artifact_dir: Path | None = None,
) -> LoggedRun:
    """Log a training run to MLflow and (if it passed the gate) register the model.

    ``register`` defaults to ``result.passed_gate`` so a below-gate model is logged
    but never registered (FR-4.3, FR-5.2).
    """
    uri = tracking_uri or default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    _ensure_experiment(uri, artifact_dir)

    should_register = result.passed_gate if register is None else register

    with mlflow.start_run() as run:
        # Parameters.
        mlflow.log_param("model", result.config.model)
        mlflow.log_param("seed", result.config.seed)
        mlflow.log_param("min_roc_auc", result.config.min_roc_auc)
        # Metrics.
        for name, value in result.metrics.to_dict().items():
            mlflow.log_metric(name, value)
        mlflow.log_metric("fairness_selection_rate_gap", result.fairness.selection_rate_gap)
        mlflow.log_metric("fairness_tpr_gap", result.fairness.tpr_gap)
        # Lifecycle evidence as tags (FR-5.2, FR-17).
        mlflow.set_tag("dataset_version", evidence.dataset_version)
        mlflow.set_tag("code_commit", evidence.code_commit)
        mlflow.set_tag("pipeline_execution_id", evidence.pipeline_execution_id)
        mlflow.set_tag("passed_gate", str(result.passed_gate))

        # Model artifact.
        model_version: str | None = None
        registered_name = REGISTERED_MODEL_NAME if should_register else None
        # Use cloudpickle serialization: MLflow 3.x defaults to skops, which refuses
        # to load tree-based sklearn models without an explicit trusted-types list.
        # These are our own artifacts produced in this pipeline (decision D-6).
        info = mlflow.sklearn.log_model(
            sk_model=estimator,
            name="model",
            registered_model_name=registered_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        if should_register:
            client = MlflowClient(tracking_uri=uri)
            versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
            if versions:
                model_version = versions[0].version
                # Persist evidence link on the registered version too.
                client.set_model_version_tag(
                    REGISTERED_MODEL_NAME,
                    model_version,
                    "code_commit",
                    evidence.code_commit,
                )
        _ = info
        return LoggedRun(
            run_id=run.info.run_id,
            model_version=model_version,
            tracking_uri=uri,
        )


def load_model(model_version: str, *, tracking_uri: str | None = None) -> Pipeline:
    """Load a specific registered model version (FR-5 AC-3)."""
    uri = tracking_uri or default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    return mlflow.sklearn.load_model(model_uri)


def latest_model_version(*, tracking_uri: str | None = None) -> str | None:
    """Return the highest registered model version, or None if none exist."""
    uri = tracking_uri or default_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except Exception:
        return None
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).version
