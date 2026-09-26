"""Tests for the Airflow DAG module (FR-6).

These verify the DAG is import-safe without Airflow (D-30) and that its task
callables are the thin wrappers over the real stage functions. When Airflow IS
installed, an extra assertion checks the task graph and the approval gate.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_dag_module_imports_without_airflow() -> None:
    """Importing the DAG module must never break the project (D-30)."""
    mod = importlib.import_module("pipelines.airflow.dags.drifttrace_dag")
    assert mod.DAG_ID == "drifttrace_pipeline"
    # build_dag returns None when Airflow is absent; a DAG object when present.
    assert hasattr(mod, "build_dag")


@pytest.mark.unit
def test_task_callables_are_wrappers() -> None:
    mod = importlib.import_module("pipelines.airflow.dags.drifttrace_dag")
    for name in [
        "_task_ingest",
        "_task_validate",
        "_task_drift_check",
        "_task_report",
        "_task_retrain",
    ]:
        assert callable(getattr(mod, name))


@pytest.mark.unit
def test_retrain_task_skips_without_approval() -> None:
    """The DAG retrain task must not retrain unless conf approves it (FR-6.2)."""
    mod = importlib.import_module("pipelines.airflow.dags.drifttrace_dag")

    class _DagRun:
        conf: dict = {}

    out = mod._task_retrain(dag_run=_DagRun())
    assert out["stage"] == "retrain"
    assert out["ok"] is True
    assert "skipped" in out["detail"]


@pytest.mark.unit
def test_dag_structure_when_airflow_present() -> None:
    """If Airflow is installed, verify the task graph and dependencies."""
    airflow = pytest.importorskip("airflow")
    assert airflow is not None
    mod = importlib.import_module("pipelines.airflow.dags.drifttrace_dag")
    dag = mod.build_dag()
    assert dag is not None
    task_ids = set(dag.task_ids)
    assert task_ids == {"ingest", "validate", "drift_check", "report", "retrain"}

    # ingest -> validate -> drift_check -> {report, retrain}
    validate = dag.get_task("validate")
    drift = dag.get_task("drift_check")
    assert "ingest" in validate.upstream_task_ids
    assert "validate" in drift.upstream_task_ids
    assert {"report", "retrain"}.issubset(drift.downstream_task_ids)
