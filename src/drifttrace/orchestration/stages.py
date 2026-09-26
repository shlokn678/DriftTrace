"""Pipeline stages for the DriftTrace DAG, as plain library functions (FR-6.4).

Each function here is a thin, deterministic, broker-free step that the CLI and the
Airflow DAG both call, so the whole pipeline runs identically with or without an
orchestrator (FR-6.4, NFR-8). The stage order mirrors the pitch:

    ingest -> validate -> drift-check -> report / retrain

Rules enforced here (independent of Airflow):
- Validation failure blocks downstream work (FR-6 AC-2): ``validate_stage`` raises
  :class:`ValidationFailed`, and the Airflow DAG maps that to a failed task that
  stops ``drift_check`` / ``report`` / ``retrain``.
- The report branch always runs after drift-check; retraining is reachable only
  through an explicit human-approval gate (FR-6.2, FR-6 AC-3): ``retrain_stage``
  refuses to run unless ``approved=True``.

Note on drift-check scope: full KS/PSI per-node detection is Phase 4 (FR-9). This
stage uses the Phase-4 detector when it is available and otherwise falls back to a
real, simple population-mean-shift comparison against the stored baseline so the DAG
branching is driven by an actual computation, never a stub. The detector is behind a
single call site so Phase 4 can substitute KS/PSI without touching the DAG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from drifttrace.config import get_paths
from drifttrace.data.build import build_dataset
from drifttrace.data.schema import load_schema
from drifttrace.data.validate import validate
from drifttrace.drift.baseline import Baseline


class ValidationFailed(RuntimeError):
    """Raised when the ingested data fails schema validation (blocks downstream)."""


class ApprovalRequired(RuntimeError):
    """Raised when retraining is attempted without explicit human approval."""


@dataclass
class StageResult:
    """Uniform result envelope for a stage."""

    stage: str
    ok: bool
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "ok": self.ok, "detail": self.detail}


# --------------------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------------------
def ingest_stage(n_rows: int = 5000, seed: int = 42, out_dir: Path | None = None) -> StageResult:
    """Ingest: generate/refresh and persist the dataset (FR-1)."""
    manifest = build_dataset(n_rows=n_rows, seed=seed, out_dir=out_dir)
    return StageResult(
        stage="ingest",
        ok=True,
        detail={
            "dataset_path": manifest["dataset_path"],
            "dataset_version": manifest["dataset_version"],
            "n_rows": manifest["n_rows"],
        },
    )


# --------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------
def validate_stage(dataset_path: Path | None = None) -> StageResult:
    """Validate the ingested dataset against the declared schema (FR-2).

    Raises :class:`ValidationFailed` if the data violates the schema, so that the
    orchestrator stops drift-check / report / retrain (FR-6 AC-2).
    """
    paths = get_paths()
    ds_path = dataset_path or (paths.data / "dataset.csv")
    frame = pd.read_csv(ds_path)
    report = validate(frame, load_schema())
    if not report.passed:
        raise ValidationFailed(json.dumps(report.to_dict()))
    return StageResult(stage="validate", ok=True, detail=report.to_dict())


# --------------------------------------------------------------------------------------
# drift-check
# --------------------------------------------------------------------------------------
def _mean_shift_verdict(reference: list[float], current: list[float], rel_threshold: float) -> dict:
    """Real, simple drift signal: relative shift of the population mean.

    Interim detector used until Phase 4 installs KS/PSI (FR-9). Deterministic and
    dependency-light. Returns the reference/current means, the relative shift, and a
    boolean verdict.
    """
    import statistics

    if not reference or not current:
        return {"drifted": False, "reason": "insufficient_data"}
    ref_mean = statistics.fmean(reference)
    cur_mean = statistics.fmean(current)
    denom = abs(ref_mean) if ref_mean != 0 else 1.0
    rel_shift = abs(cur_mean - ref_mean) / denom
    return {
        "drifted": rel_shift >= rel_threshold,
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "relative_shift": rel_shift,
        "threshold": rel_threshold,
    }


def drift_check_stage(
    current: pd.DataFrame,
    baseline_path: Path | None = None,
    rel_threshold: float = 0.2,
) -> StageResult:
    """Compare a current window against the stored baseline, per node.

    Produces genuine per-node verdicts and an overall ``drift_detected`` flag that
    the DAG uses for branching. Continuous nodes only in this interim detector;
    Phase 4 extends this to KS/PSI including categorical PSI.
    """
    paths = get_paths()
    bpath = baseline_path or (paths.artifacts / "baseline.json")
    baseline = Baseline.load(bpath)

    per_node: dict[str, dict] = {}
    drifted_nodes: list[str] = []
    for node, nb in baseline.nodes.items():
        if nb.kind != "continuous" or node not in current.columns:
            continue
        current_values = [float(v) for v in current[node].dropna().to_numpy()]
        verdict = _mean_shift_verdict(nb.values, current_values, rel_threshold)
        per_node[node] = verdict
        if verdict.get("drifted"):
            drifted_nodes.append(node)

    return StageResult(
        stage="drift-check",
        ok=True,
        detail={
            "model_version": baseline.model_version,
            "drift_detected": bool(drifted_nodes),
            "drifted_nodes": drifted_nodes,
            "per_node": per_node,
            "detector": "mean_shift_interim",  # replaced by KS/PSI in Phase 4
        },
    )


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def report_stage(drift_result: StageResult, reports_dir: Path | None = None) -> StageResult:
    """Always write a report of the drift-check outcome (FR-6 AC-3, FR-17)."""
    paths = get_paths()
    out = reports_dir or paths.reports
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "pipeline_report.json"
    payload = {"drift_check": drift_result.to_dict()}
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return StageResult(
        stage="report",
        ok=True,
        detail={
            "report_path": str(report_path),
            "drift_detected": drift_result.detail.get("drift_detected", False),
        },
    )


# --------------------------------------------------------------------------------------
# retrain (human-approved gate)
# --------------------------------------------------------------------------------------
def retrain_stage(
    approved: bool,
    approver: str | None = None,
    dataset_path: Path | None = None,
    tracking_uri: str | None = None,
) -> StageResult:
    """Retrain and register a new model version, only with explicit approval.

    FR-6.2 / FR-12.2: retraining is human-approved; the DAG must not silently push a
    new model. Raises :class:`ApprovalRequired` when ``approved`` is False.
    """
    if not approved:
        raise ApprovalRequired(
            "retraining requires explicit human approval (pass approved=True / --approve)"
        )
    # Imported lazily so importing this module does not require MLflow (NFR-8).
    from drifttrace.training.pipeline import run_training_pipeline

    result = run_training_pipeline(dataset_path=dataset_path, tracking_uri=tracking_uri)
    detail = result.to_dict()
    detail["approved_by"] = approver or "unspecified"
    return StageResult(stage="retrain", ok=True, detail=detail)
