---
inclusion: always
---
# DriftTrace — Structure Steering

- `src/drifttrace/` core library, one subpackage per component (data, features, graph, training,
  drift, rca, alerting, serving, streaming, explain, governance, cli). Keep it importable and
  unit-testable without Airflow/broker. Plus `bootstrap.py` for fresh-clone setup.
- `pipelines/airflow/dags/` thin DAG wrappers over CLI/library functions.
- `config/` declared config: graph.yaml, schema.yaml, drift.yaml, governance.yaml.
- `scripts/setup.ps1` fresh-clone bootstrap (local-first; no Docker).
- `.github/workflows/` GitHub Actions CI.
- `tests/{unit,integration,e2e}/` mirrors the testing strategy in the spec.
- `docs/` architecture, runbook, governance checklist.
- `data/ artifacts/ reports/` runtime outputs (git-ignored except .keep).
- `.kiro/specs/drifttrace/` the source-of-truth spec (requirements, design, tasks, mapping,
  decisions). Update the spec before large code changes.
