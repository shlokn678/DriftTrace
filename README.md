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

## Current implementation

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
- The current MVP uses a **scikit-learn classifier**.
- The current dependency graph is `income → credit_score → risk_score → prediction`.
- **Prometheus/Grafana** and **AWS SageMaker** are **not** part of the implemented MVP.
