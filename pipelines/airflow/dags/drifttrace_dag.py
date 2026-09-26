"""DriftTrace Airflow DAG: a thin wrapper over the orchestration stages (FR-6).

Implements the pitch pipeline ``ingest -> validate -> drift-check -> report / retrain``
with a human-approval gate before retraining (FR-6.1, FR-6.2, FR-6 AC-1..AC-3).

Design constraints honoured here:
- Every task is a thin wrapper over a CLI-callable library function in
  ``drifttrace.orchestration.stages`` (FR-6.4). No business logic lives in the DAG.
- Validation failure raises, so Airflow stops ``drift_check`` / ``report`` / ``retrain``
  (FR-6 AC-2), and ``report`` is wired to always run after drift-check while ``retrain``
  sits behind an explicit approval gate (FR-6 AC-3).
- The module is import-safe: if Airflow is not installed (e.g. a laptop without the
  ``orchestration`` extra), importing this file does not raise, so the rest of the
  project keeps working (decision D-30). Airflow discovers ``dag`` only when present.

Approval gate: the ``retrain`` task reads the ``approved`` DAG-run configuration flag
(``{"approved": true, "approver": "<name>"}``). Without it, the task short-circuits and
no new model is registered — retraining is never silent (FR-6.2).
"""

from __future__ import annotations

from typing import Any

# ---- Real stage functions (import unconditionally; they have no Airflow dependency) ---
from drifttrace.orchestration.stages import (
    ApprovalRequired,
    drift_check_stage,
    ingest_stage,
    report_stage,
    retrain_stage,
    validate_stage,
)

DAG_ID = "drifttrace_pipeline"


# ---- Task callables (thin wrappers) ---------------------------------------------------
def _task_ingest(**_: Any) -> dict:
    return ingest_stage().to_dict()


def _task_validate(**_: Any) -> dict:
    # Raises ValidationFailed on bad data -> Airflow marks the task failed and stops
    # the downstream drift_check / report / retrain tasks (FR-6 AC-2).
    return validate_stage().to_dict()


def _task_drift_check(**_: Any) -> dict:
    import pandas as pd

    from drifttrace.config import get_paths

    frame = pd.read_csv(get_paths().data / "dataset.csv")
    return drift_check_stage(frame).to_dict()


def _task_report(**context: Any) -> dict:
    import pandas as pd

    from drifttrace.config import get_paths

    frame = pd.read_csv(get_paths().data / "dataset.csv")
    drift = drift_check_stage(frame)
    return report_stage(drift).to_dict()


def _task_retrain(**context: Any) -> dict:
    # Human-approval gate (FR-6.2 / FR-12.2): only retrain when the DAG run was
    # triggered with conf {"approved": true}. Otherwise skip without registering.
    conf = (context.get("dag_run").conf if context.get("dag_run") else {}) or {}
    approved = bool(conf.get("approved", False))
    approver = conf.get("approver")
    if not approved:
        return {
            "stage": "retrain",
            "ok": True,
            "detail": {"skipped": "no approval (human-approved gate)"},
        }
    try:
        return retrain_stage(approved=True, approver=approver).to_dict()
    except ApprovalRequired as exc:  # defensive; should not happen when approved=True
        return {"stage": "retrain", "ok": False, "detail": {"error": str(exc)}}


def build_dag() -> Any:
    """Construct the Airflow DAG. Returns None if Airflow is not installed."""
    try:
        import pendulum
        from airflow import DAG

        # PythonOperator moved to the 'standard' provider in Airflow 3.x; fall back to
        # the 2.x location. Either import keeps this DAG valid across Airflow versions.
        try:
            from airflow.providers.standard.operators.python import PythonOperator
        except Exception:
            from airflow.operators.python import PythonOperator
    except Exception:
        # Airflow not installed / not importable in this environment: keep the module
        # import-safe so the rest of the project (CLI, pipeline, tests) is unaffected
        # (D-30). This is also the native-Windows case (Airflow runtime is POSIX-only).
        return None

    with DAG(
        dag_id=DAG_ID,
        description="DriftTrace: ingest -> validate -> drift-check -> report / retrain",
        schedule=None,
        start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
        catchup=False,
        tags=["drifttrace", "mlops"],
    ) as dag:
        ingest = PythonOperator(task_id="ingest", python_callable=_task_ingest)
        validate = PythonOperator(task_id="validate", python_callable=_task_validate)
        drift_check = PythonOperator(task_id="drift_check", python_callable=_task_drift_check)
        report = PythonOperator(task_id="report", python_callable=_task_report)
        retrain = PythonOperator(task_id="retrain", python_callable=_task_retrain)

        # ingest -> validate -> drift_check -> {report, retrain}
        # report always runs after drift_check; retrain is gated inside _task_retrain.
        ingest >> validate >> drift_check >> [report, retrain]

    return dag


# Airflow's DagBag imports this module and looks for a module-level DAG object.
dag = build_dag()
