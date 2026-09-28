# DriftTrace

**An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift**

## What it does

DriftTrace watches a machine-learning pipeline in production and finds *why* it starts
to fail, not just *that* it failed. It:

- monitors an ML pipeline for data drift
- compares production data against the training baseline
- detects drift per feature using the **KS-test** and **PSI**
- reads a declared **dependency graph** (via **NetworkX**) and traces drift upstream
- identifies the **root cause** and marks downstream nodes as **symptoms**
- produces root-cause alerts and monitoring reports
- provides **SHAP / LIME** explanations for predictions
- supports **operator-approved** rollback and retraining (never automatic)

Instead of firing a separate alarm for every drifted feature, DriftTrace turns a wall of
alerts into one actionable diagnosis.

## Demo pipeline

```
income
   ↓
credit_score
   ↓
risk_score
   ↓
ML model
   ↓
default / not default
```

## Main technologies

- Python
- scikit-learn
- FastAPI
- Docker / Docker Compose
- MLflow
- DVC
- Apache Airflow
- NetworkX
- SciPy
- SHAP / LIME
- Redpanda
- React + TypeScript + Vite

## Project structure

```
src/         core Python library (data, features, drift, rca, serving, streaming, ...)
frontend/    React + TypeScript dashboard
tests/       unit / integration / e2e tests
config/      declared config (dependency graph, schema, drift, governance)
docker/      Dockerfile + docker-compose
pipelines/   Airflow DAG
docs/        architecture and runbook
scripts/     helper scripts
data/        generated dataset (runtime)
artifacts/   model baseline and run artifacts (runtime)
reports/     monitoring reports and alerts (runtime)
```

- **src/** holds the real logic: drift detection, root-cause analysis, and the FastAPI service.
- **frontend/** is the monitoring dashboard.
- **config/** defines the dependency graph and thresholds.
- **docker/** runs the whole stack locally.

## Setup

**Requirements:** Python, Node.js / npm, Docker Desktop.

Clone:

```
git clone https://github.com/shlokn678/DriftTrace.git
cd DriftTrace
```

Start the backend stack:

```
$env:DRIFTTRACE_USE_REDPANDA = "true"
docker compose -f docker/docker-compose.yml --profile full up -d
```

Check the containers:

```
docker ps --filter "name=drifttrace" --format "{{.Names}} | {{.Status}}"
```

- API: http://localhost:8000
- Health: http://localhost:8000/health

Start the frontend in a second terminal:

```
cd frontend
npm install
npm run dev
```

- Dashboard: http://localhost:5173

## Main demo

The main scenario simulates an income distribution change (monthly income read as
annual). All three chained features drift, but DriftTrace follows the dependency graph
and reports:

- **income → ROOT CAUSE**
- **credit_score → SYMPTOM**
- **risk_score → SYMPTOM**

Rather than treating the three as separate alerts, it traces the drift upstream and
identifies `income` as the single root cause, with the others as downstream symptoms.

## Model-agnostic design

DriftTrace separates *model-specific prediction* from the *monitoring and root-cause
engine*. A model plugs in through a thin **adapter** that turns its raw inputs and outputs
into standardized prediction events; the drift, dependency-graph, and RCA core never sees
model internals.

```
USER MODEL → MODEL ADAPTER → STANDARDIZED PREDICTION EVENTS
          → DRIFTTRACE CORE → KS / PSI → DEPENDENCY GRAPH + RCA
          → ALERTS / REPORTS / EXPLANATIONS
```

The adapter layer lives in `src/drifttrace/adapters/` (`base`, `sklearn_adapter`,
`registry`). The MVP ships **one** adapter — **scikit-learn** — and the boundary is
designed so additional frameworks can be added without touching the core.

## Model onboarding

Upload one model file; DriftTrace inspects it and asks only for what it cannot detect.

1. **Upload** a model file (`.pkl`, `.pickle`, `.joblib`) — drag-and-drop or browse in the
   dashboard, or `POST /models/upload`.
2. **Inspect** — DriftTrace detects the framework, name, task, features, and whether the
   model exposes probabilities.
3. **Reference data** — reused automatically when a baseline already exists; requested only
   if missing.
4. **Dependency graph** — reused/detected from `config/graph.yaml`; requested only if
   missing. Missing dependencies do not block monitoring, they only limit upstream RCA.
5. **Start monitoring** — when the model is supported and a reference exists, it is marked
   **ready to monitor**. Nothing is fabricated.

The dashboard opens on a **simplified view** — model, system status, drift status, root
cause, and affected features — with KS/PSI evidence, the dependency graph, SHAP/LIME, the
operator console, and system health available behind **View details**.

## Current implementation

- [x] Model-agnostic adapter layer (scikit-learn adapter)
- [x] Minimal-input model onboarding (upload → inspect → ready)
- [x] Simplified dashboard with progressive disclosure
- [x] Data generation and validation
- [x] DVC data versioning
- [x] Model training and evaluation
- [x] MLflow tracking and model registry
- [x] Airflow orchestration
- [x] FastAPI model serving
- [x] Docker / Docker Compose deployment
- [x] Prediction event streaming
- [x] KS/PSI drift detection
- [x] NetworkX root-cause analysis
- [x] Root-cause alerting
- [x] SHAP/LIME explanations
- [x] Fairness and privacy checks
- [x] React monitoring dashboard

## Current scope

- The current demo uses a **synthetic loan-default dataset**.
- The monitoring/RCA core is **model-agnostic** through the adapter layer; the only
  implemented adapter is **scikit-learn**.
- The current dependency graph is `income → credit_score → risk_score → prediction`.
- **Prometheus/Grafana** and **AWS SageMaker** are **not** part of the implemented MVP.
