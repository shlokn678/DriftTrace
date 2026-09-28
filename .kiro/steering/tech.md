---
inclusion: always
---
# DriftTrace — Tech Steering

Stack: FastAPI + GitHub Actions, with Git, NetworkX, scikit-learn, SciPy (KS-test), and
SHAP/LIME. Runtime is local-first: a Python virtualenv for the backend and Node/Vite for the
dashboard. No Docker. A Kafka-API broker (Redpanda) is optional and separately-run; it is NOT
required for the normal local/demo workflow. Models are uploaded as bundles rather than
trained here, so the loan training pipeline, MLflow model registry, DVC dataset tracking, and
the Airflow DAG were removed with the loan domain (they can return as generic tooling later).
Stretch: Prometheus/Grafana and AWS SageMaker. Language: Python 3.11+.

Principles:
- Keep drift, graph, and RCA logic as pure, deterministic, broker-free libraries. FastAPI,
  Airflow, and an optional broker are thin adapters over them.
- The core is model-agnostic through a model adapter layer; scikit-learn is the only
  implemented adapter (classification + regression + Pipeline). The generic core never
  touches framework-specific model objects; it consumes standardized prediction events.
- Models are UPLOADED, not trained by DriftTrace: a bundle = `model.pkl` + `reference.csv`
  (+ optional `graph.json`). There is no built-in/default/fallback model. A fresh install
  starts with no active model.
- The reference data defines the feature schema and drift baseline. The optional dependency
  graph (declared, never learned) enables RCA; without it, drift detection still works but
  root-cause tracing is unavailable (not an error).
- Monitoring is asynchronous; prediction latency must never depend on drift computation.
- Alert only on the root-cause candidate; record symptoms without alerting.
- Paths resolve from the project root (or DRIFTTRACE_* env overrides); never hard-code
  machine-specific absolute paths. A fresh clone reconstructs runtime dirs via the bootstrap
  (which trains/registers nothing).
- Never invent credentials, cloud accounts, datasets, or external services. The API is
  unauthenticated and localhost/internal only.
- Enforce lint/format/type/tests in CI. Prefer direct local execution for laptop dev.
