---
inclusion: always
---
# DriftTrace — Tech Steering

MVP stack: FastAPI + MLflow + GitHub Actions, with Git + DVC, Apache Airflow (optional),
NetworkX, scikit-learn, SciPy (KS-test), and SHAP/LIME. Runtime is local-first: a Python
virtualenv for the backend and Node/Vite for the dashboard. No Docker. A Kafka-API broker
(Redpanda) is optional and separately-run; it is NOT required for the normal local/demo
workflow. Stretch: Prometheus/Grafana and AWS SageMaker. Language: Python 3.11+.

Principles:
- Keep drift, graph, and RCA logic as pure, deterministic, broker-free libraries. FastAPI,
  Airflow, and an optional broker are thin adapters over them.
- The core is model-agnostic through a model adapter layer; scikit-learn is the only
  implemented adapter. The generic core never touches framework-specific model objects.
- Monitoring is asynchronous; prediction latency must never depend on drift computation.
- Alert only on the root-cause candidate; record symptoms without alerting.
- The dependency graph is declared (YAML -> NetworkX), never learned.
- Reproducibility: every run yields five lifecycle-evidence artifacts (dataset version, code
  commit, pipeline execution, model artifact, monitoring report). Record seeds. A fresh clone
  reconstructs all runtime state deterministically via the bootstrap (no copied artifacts).
- Paths resolve from the project root (or DRIFTTRACE_* env overrides); never hard-code
  machine-specific absolute paths.
- Never invent credentials, cloud accounts, datasets, or external services. DVC remote and
  MLflow backend are local. The MVP API is unauthenticated and localhost/internal only.
- Enforce lint/format/type/tests in CI. Prefer direct local execution for laptop dev.
