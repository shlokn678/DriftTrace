---
inclusion: always
---
# DriftTrace — Tech Steering

MVP stack (pitch): FastAPI + Docker Compose + MLflow + GitHub Actions, with Git + DVC, Apache
Airflow, NetworkX, scikit-learn, SciPy (KS-test), SHAP/LIME, and a Kafka-API broker (Redpanda).
Stretch: Prometheus/Grafana and AWS SageMaker. Language: Python 3.11.

Principles:
- Keep drift, graph, and RCA logic as pure, deterministic, broker-free libraries. Airflow,
  Kafka, FastAPI, and Docker are thin adapters over them.
- Monitoring is asynchronous; prediction latency must never depend on drift computation.
- Alert only on the root-cause candidate; record symptoms without alerting.
- The dependency graph is declared (YAML -> NetworkX), never learned.
- Reproducibility: every run yields five lifecycle-evidence artifacts (dataset version, code
  commit, pipeline execution, model artifact, monitoring report). Record seeds.
- Never invent credentials, cloud accounts, datasets, or external services. DVC remote and
  MLflow backend are local. The MVP API is unauthenticated and localhost/internal only.
- Enforce lint/format/type/tests in CI. Prefer the `core` Compose profile for laptop dev.
