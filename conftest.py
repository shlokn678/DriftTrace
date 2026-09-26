"""Pytest configuration: ensure the repo root is importable.

The package lives under ``src/`` (installed editable), but the Airflow DAG lives at
``pipelines/airflow/dags/`` at the repo root. Adding the repo root to ``sys.path``
lets tests import ``pipelines.airflow.dags.drifttrace_dag`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
