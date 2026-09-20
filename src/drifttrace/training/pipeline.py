"""Training pipeline orchestration as a plain library function (FR-6.4).

Reads the materialized dataset, trains + evaluates, builds the versioned drift
baseline, logs the run to MLflow, and registers the model when it passes the gate.
This function is called by the DVC ``train`` stage, the CLI, and the Airflow DAG,
so the whole pipeline runs identically with or without an orchestrator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drifttrace.config import get_paths
from drifttrace.drift.baseline import build_baseline
from drifttrace.features.transform import fit_params, transform
from drifttrace.graph.loader import load_graph
from drifttrace.training.evidence import LifecycleEvidence, dataset_version, git_commit, new_run_id
from drifttrace.training.train import TrainConfig, train


@dataclass
class PipelineResult:
    """Summary of a full training pipeline run."""

    run_id: str
    model_version: str | None
    passed_gate: bool
    roc_auc: float
    evidence: LifecycleEvidence

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "passed_gate": self.passed_gate,
            "roc_auc": self.roc_auc,
            "evidence": self.evidence.to_dict(),
        }


def run_training_pipeline(
    dataset_path: Path | None = None,
    config: TrainConfig | None = None,
    tracking_uri: str | None = None,
    artifacts_dir: Path | None = None,
) -> PipelineResult:
    """Run train -> baseline -> log/register and persist a run manifest + baseline."""
    # Imported lazily so the core library does not hard-require MLflow (NFR-8).
    from drifttrace.training.registry import log_training_run

    paths = get_paths()
    ds_path = dataset_path or (paths.data / "dataset.csv")
    art_dir = artifacts_dir or paths.artifacts
    art_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(ds_path)
    csv_bytes = ds_path.read_bytes()

    estimator, result = train(frame, config)

    evidence = LifecycleEvidence(
        dataset_version=dataset_version(csv_bytes),
        code_commit=git_commit(),
        pipeline_execution_id=new_run_id(),
    )

    logged = log_training_run(estimator, result, evidence, tracking_uri=tracking_uri)
    evidence.model_version = logged.model_version

    # Build and persist the versioned drift baseline for this model version (FR-9.3).
    graph = load_graph()
    featured = transform(frame, fit_params(frame["income"].to_numpy(dtype=float)))
    baseline = build_baseline(
        featured, graph, model_version=(logged.model_version or "unregistered")
    )
    baseline.save(art_dir / "baseline.json")

    manifest = PipelineResult(
        run_id=logged.run_id,
        model_version=logged.model_version,
        passed_gate=result.passed_gate,
        roc_auc=result.metrics.roc_auc,
        evidence=evidence,
    )
    (art_dir / "run_manifest.json").write_text(
        json.dumps({**manifest.to_dict(), "metrics": result.metrics.to_dict()}, indent=2),
        "utf-8",
    )
    return manifest
