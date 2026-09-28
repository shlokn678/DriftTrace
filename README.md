# DriftTrace

**An intelligent, model-agnostic ML monitoring and root-cause diagnosis system.**

## What it does

DriftTrace watches a deployed model and finds *why* it starts to drift, not just *that*
it did. It:

- detects data drift per feature using the **KS-test** and **PSI**
- identifies which features are affected
- when a dependency graph is provided, traces upstream through it (**NetworkX**) to the
  earliest drifted node — the **root-cause candidate** — and marks downstream drift as
  **symptoms**
- turns related drift signals into one focused **incident / alert**
- provides **SHAP / LIME** explanations for predictions
- supports **operator-approved** actions (never automatic)

Instead of firing a separate alarm for every drifted feature, DriftTrace turns a wall of
alerts into one actionable diagnosis and lets a human decide the response.

## Model-agnostic by design

You upload a **model bundle**; DriftTrace inspects it, you activate it, and its incoming
predictions become monitoring data. There is **no built-in, default, or fallback model** —
a fresh install starts with no model active.

```
USER MODEL BUNDLE
    -> MODEL ADAPTER (scikit-learn)
    -> STANDARDIZED PREDICTION EVENT
    -> DRIFTTRACE CORE -> KS / PSI
    -> OPTIONAL DEPENDENCY GRAPH + RCA
    -> INCIDENT / REPORT / EXPLANATION
    -> HUMAN OPERATOR
```

The adapter layer (`src/drifttrace/adapters/`) is the only thing that touches the model.
The MVP implements **one adapter — scikit-learn** (classifiers, regressors, and
`Pipeline` objects). XGBoost / LightGBM / ONNX / PyTorch are possible future adapters and
are **not** implemented.

## The model bundle (three files)

Upload a single `.zip` containing:

```
my_model.drift.zip
├── model.pkl        (required)  the trained scikit-learn model or Pipeline
├── reference.csv    (required)  the reference/baseline data for that model
└── graph.json       (optional)  feature dependency edges for stronger RCA
```

- **model.pkl** — any supported scikit-learn estimator or `Pipeline` (preprocessing +
  estimator is one deployable model; DriftTrace does not recreate preprocessing). `.joblib`
  and `.pickle` are also accepted.
- **reference.csv** — defines the feature schema and the drift baseline for *that* model.
  Each model has its own reference data. Numeric columns get KS/PSI; categorical columns
  get frequency PSI.
- **graph.json** *(optional)* — dependency edges, e.g.
  `{"edges": [["feature_a", "feature_b"], ["feature_b", "feature_c"]]}`. When present, RCA
  traces drift upstream. When absent, drift detection still works but root-cause tracing is
  unavailable (a normal state, not an error).

DriftTrace derives framework, model name, task (classification/regression), feature list,
and probability support automatically. It only asks for genuinely missing information.

## Main technologies

Python, scikit-learn, FastAPI, NetworkX, SciPy, SHAP / LIME, React + TypeScript + Vite.
Runtime is **local-first**: a Python virtualenv for the backend and Node/Vite for the
dashboard. **No Docker.** An optional, separately-run Redpanda broker is supported but not
required.

## Prerequisites

- **Python 3.11+** (`requires-python >= 3.11`)
- **Node.js / npm** (for the dashboard)
- **Git**

No Docker, no message broker, and no cloud account are required.

## Fresh clone -> running

```powershell
# 1. Clone
git clone https://github.com/shlokn678/DriftTrace.git
cd DriftTrace

# 2. Set up (creates .venv, installs the project, prepares runtime dirs). One command.
#    It does NOT create or train any model.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1

# 3. Start the backend API
.\.venv\Scripts\python.exe -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000

# 4. Start the dashboard (second terminal)
cd frontend
npm run dev
```

Open the dashboard at **http://localhost:5173** (API at **http://localhost:8000**). On
first open, **no model is active** — upload a bundle to begin.

On Linux/macOS use `bash scripts/setup.sh` for step 2 and
`./.venv/bin/python -m drifttrace.cli.main serve ...` for step 3.

## Using DriftTrace

### 1. Upload and activate a model

In the dashboard, drop your bundle `.zip` onto **Add Model** (or browse). DriftTrace
inspects it and shows the framework, task, features, reference status, and whether a graph
was provided. Click **Use this model** to make it the **active model** — `/predict` now
serves it. **Deactivate** clears it (there is no fallback).

Via the API directly:

```powershell
# Upload a bundle .zip and inspect it
curl.exe -F "file=@my_model.drift.zip" http://localhost:8000/models/upload
# Activate the returned model id
curl.exe -X POST http://localhost:8000/models/<model_id>/activate
# See the active model (or the explicit no-model state)
curl.exe http://localhost:8000/models/active
```

### 2. Predict

Predictions use the active model's own feature names (no domain-specific fields):

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"features\": {\"feature_a\": 1.2, \"feature_b\": 5, \"feature_c\": 0.3}}'
```

Each successful prediction emits a standardized event that feeds monitoring.

### 3. Run a drift test

A generic drift test perturbs a sample of the active model's reference data and runs it
through the real KS/PSI + RCA pipeline (no fabricated results):

```powershell
curl.exe -X POST http://localhost:8000/demo/run-drift-test `
  -H "Content-Type: application/json" `
  -d '{\"intensity\": 2.0, \"n\": 300, \"seed\": 7}'
```

In the dashboard, use **Run Drift Test** under **View Details**. With a dependency graph,
the earliest drifted feature is reported as the root cause and downstream features as
symptoms; without a graph, the affected features are listed and the origin is
"undetermined".

### 4. Check the diagnosis

```powershell
curl.exe http://localhost:8000/rca/latest      # latest persisted monitoring/RCA report
```

The dashboard's default view shows the active model, system status, drift status, root
cause, and affected features. Technical detail (KS/PSI, dependency graph, SHAP/LIME,
metrics) is behind **View Details**.

## Project structure

```
src/drifttrace/
  bundle/     model bundle: loader (zip/dir), reference profiling, graph.json parsing
  adapters/   model-agnostic adapter layer (scikit-learn adapter)
  drift/      KS/PSI engine + baseline + config
  graph/      NetworkX dependency-graph DAG
  rca/        root-cause analysis
  streaming/  prediction-event schema + windowing + monitor + file source
  serving/    FastAPI app + onboarding/active-model registry
  explain/    SHAP / LIME
  governance/ audit + privacy + operator (approval) + report
  bootstrap.py  fresh-clone setup (runtime dirs only; trains no model)
frontend/     React + TypeScript + Vite dashboard
config/       drift.yaml (thresholds) + governance.yaml (PII / fairness config)
scripts/      setup.ps1 / setup.sh + dev helpers
docs/         architecture, runbook, governance checklist
data/ artifacts/ reports/   runtime outputs (git-ignored; artifacts/uploaded_models holds bundles)
```

## Troubleshooting

- **`Python 3.11+ is required`** — install Python 3.11+; on Windows the `py` launcher
  (`py -3.13`) is used automatically if present.
- **`/ready` returns `ready: false`** — no model is active. Upload and activate a bundle.
- **Dashboard shows values as unavailable** — the API is not reachable on port 8000; the
  dashboard proxies `/api` to it in dev.
- **`/predict` returns 503** — no active model. Activate a bundle first.

## Limitations

- Only the **scikit-learn** adapter is implemented (classification, regression, Pipeline).
- The onboarding/active-model registry is **in-memory** per process; there is no database.
- Reference data must contain the model's feature columns; categorical drift uses frequency
  comparison.
- Fairness analysis is unavailable unless a sensitive attribute is declared in
  `config/governance.yaml` (never inferred).
- DriftTrace does not train or retrain models; retraining happens outside DriftTrace and you
  upload the new bundle.
- **Prometheus/Grafana**, **AWS SageMaker**, graph learning, and temporal GNNs are out of
  scope. Redpanda is optional and not required.
