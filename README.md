# DriftTrace

**An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift**

## What it does

DriftTrace watches a machine-learning pipeline in production and finds *why* it starts
to fail, not just *that* it failed. It:

- monitors an ML pipeline for data drift
- compares production data against a training baseline
- detects drift per feature using the **KS-test** and **PSI**
- reads a declared **dependency graph** (via **NetworkX**) and traces drift upstream
- identifies the **root cause** and marks downstream nodes as **symptoms**
- produces root-cause alerts and monitoring reports
- provides **SHAP / LIME** explanations for predictions
- supports **operator-approved** rollback and retraining (never automatic)

Instead of firing a separate alarm for every drifted feature, DriftTrace turns a wall of
alerts into one actionable diagnosis.

## Model-agnostic design

DriftTrace separates *model-specific prediction* from the *monitoring and root-cause
engine*. A model plugs in through a thin **adapter** that turns its inputs and outputs
into standardized prediction events; the drift, dependency-graph, and RCA core never sees
model internals.

```
USER MODEL -> MODEL ADAPTER -> STANDARDIZED PREDICTION EVENTS
           -> DRIFTTRACE CORE -> KS / PSI -> DEPENDENCY GRAPH + RCA
           -> ALERTS / REPORTS / EXPLANATIONS
```

The adapter layer lives in `src/drifttrace/adapters/`. The MVP ships **one** adapter,
**scikit-learn**, and the boundary is designed so other frameworks can be added without
touching the core.

## Demo pipeline

```
income -> credit_score -> risk_score -> ML model -> default / not default
```

## Main technologies

Python, scikit-learn, FastAPI, MLflow, DVC, Apache Airflow (optional), NetworkX, SciPy,
SHAP / LIME, React + TypeScript + Vite. Runtime is **local-first**: a Python virtualenv
for the backend and Node/Vite for the dashboard. **No Docker.**

## Prerequisites

- **Python 3.11+** (the project's `requires-python` is `>=3.11`)
- **Node.js / npm** (for the dashboard)
- **Git**

No Docker, no message broker, and no cloud account are required.

## Fresh clone -> running in 4 steps

```powershell
# 1. Clone
git clone https://github.com/shlokn678/DriftTrace.git
cd DriftTrace

# 2. Set up + bootstrap (creates .venv, installs the project, generates the dataset,
#    trains + registers the loan model, writes the drift baseline). One command:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1

# 3. Start the backend API
.\.venv\Scripts\python.exe -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000

# 4. Start the dashboard (second terminal)
cd frontend
npm run dev
```

Then open the dashboard at **http://localhost:5173** (API at **http://localhost:8000**).

On Linux/macOS use `bash scripts/setup.sh` for step 2 and
`./.venv/bin/python -m drifttrace.cli.main serve ...` for step 3.

### What the setup does

`scripts\setup.ps1` (or the cross-platform `python -m drifttrace.bootstrap`):
creates the `.venv`, installs the project with the `serving,tracking,streaming` extras,
then reconstructs all runtime state deterministically from tracked files - runtime
directories, the synthetic dataset, schema validation, the trained + registered model,
and the drift baseline. It is safe to re-run; pass `-Force` to rebuild from scratch. It
fails with a clear message if a prerequisite (e.g. Python 3.11+) is missing.

## Using DriftTrace

### Upload and activate a model

1. In the dashboard, drop a model file (`.pkl`, `.pickle`, `.joblib`) onto **Add Model**,
   or browse for one.
2. DriftTrace inspects it automatically (framework, name, task, features, probability
   support) and reuses your existing reference data and dependency graph. It only asks
   for genuinely missing information.
3. Click **Use this model** to make it the **active model**. `/predict` now serves it.
   The default loan model remains the fallback - **Deactivate** restores it.

The default (unauthenticated, localhost) API also exposes this directly:

```powershell
# Upload (multipart) and inspect
curl.exe -F "file=@model.pkl" http://localhost:8000/models/upload
# Activate a returned model id
curl.exe -X POST http://localhost:8000/models/<model_id>/activate
# See the active model (custom or default loan model)
curl.exe http://localhost:8000/models/active
```

### Run the drift demo (no Docker, no broker)

The main scenario simulates an income distribution change (monthly income read as
annual). All three chained features drift, but DriftTrace follows the dependency graph
and reports **income -> ROOT CAUSE**, **credit_score -> SYMPTOM**, **risk_score ->
SYMPTOM**. Run it through the real Phase 4 monitoring/RCA pipeline:

```powershell
curl.exe -X POST http://localhost:8000/demo/run-scenario `
  -H "Content-Type: application/json" `
  -d '{\"scenario\": \"income_annual\", \"n\": 300, \"seed\": 7}'
```

Other scenarios: `control` (no drift), `mid_chain` (credit_score root), `two_roots`.
In the dashboard, use the operator controls under **View Details** to run a scenario.

### Check the root-cause result

```powershell
curl.exe http://localhost:8000/rca/latest      # latest persisted RCA report
```

Or open the dashboard: the simplified view shows the active model, system status, drift
status, root cause, and affected features. Everything technical (KS/PSI, per-feature
drift, dependency graph, SHAP/LIME, governance, metrics) is behind **View Details**.

## Project structure

```
src/         core Python library (adapters, data, features, drift, rca, serving, ...)
             plus bootstrap.py (fresh-clone setup)
frontend/    React + TypeScript dashboard
tests/       unit / integration / e2e tests
config/      declared config (dependency graph, schema, drift, governance)
scripts/     setup.ps1 / setup.sh (bootstrap) + dev helpers
pipelines/   Airflow DAG (optional)
docs/        architecture and runbook
data/ artifacts/ reports/   runtime outputs (git-ignored; rebuilt by the bootstrap)
```

## Troubleshooting

- **`Python 3.11+ is required`** - install Python 3.11 or newer; on Windows the `py`
  launcher (`py -3.13`) is used automatically if present.
- **`/ready` returns `ready: false`** - the model is not loaded. Run the bootstrap:
  `.\.venv\Scripts\python.exe -m drifttrace.bootstrap` (add `--force` to rebuild).
- **Dashboard shows values as unavailable** - the API is not reachable. Confirm it is
  running on port 8000; the dashboard proxies `/api` to it in dev.
- **Rebuild everything from scratch** - `scripts\setup.ps1 -Force` (regenerates the
  dataset and retrains the model).

## Current implementation

- [x] Model-agnostic adapter layer (scikit-learn adapter)
- [x] Model onboarding + activation (upload -> inspect -> use this model -> active)
- [x] Simplified dashboard with progressive disclosure
- [x] Local-first setup + fresh-clone bootstrap (no Docker)
- [x] Data generation and validation
- [x] DVC data versioning
- [x] Model training and evaluation
- [x] MLflow tracking and model registry (local SQLite)
- [x] Airflow orchestration (optional)
- [x] FastAPI model serving
- [x] Prediction event streaming (local file; optional Redpanda)
- [x] KS/PSI drift detection
- [x] NetworkX root-cause analysis
- [x] Root-cause alerting
- [x] SHAP/LIME explanations
- [x] Fairness and privacy checks
- [x] React monitoring dashboard

## Current scope

- The demo uses a **synthetic loan-default dataset**.
- The monitoring/RCA core is **model-agnostic** through the adapter layer; the only
  implemented adapter is **scikit-learn**.
- The dependency graph is `income -> credit_score -> risk_score -> prediction`.
- The normal workflow is broker-free. **Redpanda** is optional and separately-run; it is
  not required. **Prometheus/Grafana** and **AWS SageMaker** are not part of the MVP.
