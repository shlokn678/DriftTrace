"""Construct and inspect the real Airflow DAG object.

Shims the single POSIX-only hook (os.register_at_fork) that blocks Airflow's imports
on native Windows, then builds the DriftTrace DAG via its own build_dag() and checks
the task graph and dependencies. On Linux/WSL/containers the shim is a harmless no-op.
"""

from __future__ import annotations

import os
import sys

# Import concurrent.futures BEFORE anything else so ThreadPoolExecutor is fully
# initialized (avoids a Windows-only partial-init clash between Airflow and pandas).
import concurrent.futures  # noqa: E402,F401

# POSIX-only shim so Airflow imports on native Windows (no effect on POSIX).
if not hasattr(os, "register_at_fork"):
    os.register_at_fork = lambda **kwargs: None  # type: ignore[attr-defined]


def _load_dag_module(dag_file: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("drifttrace_dag_under_test", dag_file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_heavy_optional_deps() -> None:
    """Stub pandas so DAG *construction* runs against Airflow only.

    The DAG's task callables import pandas at run time, but building the DAG object
    (operators + dependencies) does not need it. On native Windows, importing pandas
    through Airflow's import chain hits a partial-init clash, so we stub it here purely
    to validate the DAG graph. This does not affect the real runtime on Linux/WSL.
    """
    import types

    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.read_csv = lambda *a, **k: None  # type: ignore[attr-defined]
        pandas_stub.DataFrame = object  # type: ignore[attr-defined]
        sys.modules["pandas"] = pandas_stub


def main() -> int:
    import airflow  # noqa: F401  (import validates the install)

    _stub_heavy_optional_deps()
    dag_file = sys.argv[1] if len(sys.argv) > 1 else "drifttrace_dag.py"
    dag_module = _load_dag_module(dag_file)
    DAG_ID = dag_module.DAG_ID
    build_dag = dag_module.build_dag

    dag = build_dag()
    if dag is None:
        print("DAG_BUILD_RETURNED_NONE (airflow import failed inside build_dag)")
        return 1

    task_ids = set(dag.task_ids)
    print("AIRFLOW_VERSION=", airflow.__version__)
    print("DAG_ID=", DAG_ID)
    print("TASKS=", sorted(task_ids))

    expected = {"ingest", "validate", "drift_check", "report", "retrain"}
    assert task_ids == expected, f"unexpected tasks: {task_ids}"

    validate = dag.get_task("validate")
    drift = dag.get_task("drift_check")
    assert "ingest" in validate.upstream_task_ids
    assert "validate" in drift.upstream_task_ids
    assert {"report", "retrain"}.issubset(drift.downstream_task_ids)

    print("EDGES ingest->validate->drift_check->{report,retrain} OK")
    print("DAG_PARSE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
