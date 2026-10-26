# DriftTrace

**An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift**

DriftTrace monitors a deployed machine-learning model, detects distribution changes in incoming data, and uses a declared dependency graph to help identify likely root-cause candidates and downstream symptoms. It supports batch/on-demand monitoring as well as continuous event-stream monitoring, with a React dashboard for diagnostics and incident history.

> **Detect drift → Trace dependencies → Identify likely root causes → Show symptoms → Let a human decide what to do**

DriftTrace is model-agnostic across its supported **scikit-learn estimators and Pipelines**. A fresh checkout does not include a built-in production model: you upload a model bundle or generate the local test bundles.

## Contents

- [Project overview](#project-overview)
- [Features](#features)
- [Architecture and data flow](#architecture-and-data-flow)
- [Model bundle format](#model-bundle-format)
- [Installation](#installation)
- [Run locally without Redpanda](#run-locally-without-redpanda)
- [Run continuous monitoring with Redpanda](#run-continuous-monitoring-with-redpanda)
- [API endpoints](#api-endpoints)
- [Multiple root-cause candidates](#multiple-root-cause-candidates)
- [Testing and validation](#testing-and-validation)
- [Project structure](#project-structure)
- [Concepts](#concepts)
- [Limitations and future work](#limitations-and-future-work)

## Project overview

A model may behave differently after deployment because the data reaching it changes. This is commonly called **data drift**. A drift detector can flag that distributions changed, but an operator still needs to work out which affected feature is most upstream and which changes may be downstream symptoms.

DriftTrace combines:

- Reference-based statistical drift detection using the Kolmogorov–Smirnov (KS) test and Population Stability Index (PSI).
- Numerical and categorical feature monitoring, plus prediction/output monitoring when sufficient observations are available.
- An optional directed dependency graph for upstream tracing and downstream symptom identification.
- Batch/on-demand and continuous live-monitoring paths using the same drift and RCA engines.
- Incident tracking across consecutive drifted and stable windows.
- Local prediction explanations using SHAP and LIME.
- Operator-reviewed retraining and rollback workflows.
- Model/session-scoped monitoring state, so a previous model's live results are not displayed as the current model's results.

**Drift detection and root-cause analysis are separate steps.** DriftTrace first measures which distributions changed. When a dependency graph is supplied, it uses the declared edges and observed drift to help trace likely upstream root-cause candidates. This is diagnostic evidence, not proof of physical causality.

## Features

### Model onboarding

Upload a `.drift.zip` bundle containing a serialized supported scikit-learn model and its reference data. DriftTrace inspects the model, feature schema, task type, and relevant prediction capabilities before it is activated.

### Statistical drift detection

- **KS test:** reports the KS statistic and p-value for numerical distributions.
- **PSI:** measures the shift between current and reference populations.
- **Categorical drift:** compares category frequencies where applicable.
- **Output monitoring:** monitors prediction/output changes when enough output observations are available.

The current UI shows a KS p-value threshold of `p < 0.05` and a PSI drift threshold of `PSI >= 0.20`. These are configured detection rules, not universal guarantees that a model is or is not accurate.

### Dependency-aware root-cause analysis

When `graph.json` is provided, DriftTrace considers actual graph edges to distinguish affected upstream candidates from downstream symptoms. The dependency graph visualization renders arrows only for declared edges; it does not infer a connection merely because two nodes happen to appear next to each other in the UI.

### Continuous live monitoring

Live monitoring consumes standardized prediction events, groups them into non-overlapping tumbling windows, evaluates each completed window with the existing KS/PSI and RCA engines, records the results, and updates the dashboard.

Supported sources:

- **Broker-free local source:** suitable for quick local development; no Redpanda installation is required.
- **Redpanda/Kafka-compatible source:** a real broker-backed stream, used for the live demo described below.

### Incident lifecycle

The live incident tracker groups consecutive drifted windows into a continuing incident rather than opening a new incident for every window. By default:

- `DRIFTTRACE_LIVE_INCIDENT_OPEN_AFTER=2`: open after two consecutive drifted windows.
- `DRIFTTRACE_LIVE_INCIDENT_CLOSE_AFTER=3`: close after three consecutive stable windows.

A window's drift verdict still comes from the drift detector; the incident policy controls when a sequence of verdicts is treated as an incident.

### Prediction explanations

SHAP and LIME provide local explanations for an individual prediction. They answer a different question from RCA:

- **RCA:** Where might the observed drift have started in the declared dependency graph?
- **SHAP/LIME:** Which input features influenced this particular model prediction?

Explanation requests have cancellation, timeout handling, and a retry/error state. Feature contribution is not the same as causal root cause.

### Human-in-the-loop operations

DriftTrace presents evidence for investigation. Production-changing actions such as retraining or rollback remain subject to operator review; the system does not silently replace the active model.

## Architecture and data flow

```text
Model bundle
  ├── model.pkl
  ├── reference.csv
  └── graph.json (optional)
          │
          ▼
   Active model / API
          │
          ├── Batch or on-demand drift check
          │
          └── Prediction events
                   │
                   ▼
          Local event log or Redpanda topic
                   │
                   ▼
          Live monitoring consumer
                   │
                   ▼
          Tumbling windows (demo configured to 200 events)
                   │
                   ▼
          Existing KS + PSI drift engine
                   │
                   ▼
          Existing dependency-graph RCA engine
                   │
                   ▼
          Incident lifecycle + persistent history
                   │
                   ▼
          FastAPI endpoints → React dashboard
```

**Role of Docker and Redpanda:** Docker runs Redpanda as a container. Redpanda acts as the event broker that receives and distributes producer messages. DriftTrace's backend consumes those messages and analyzes them. Docker does not perform drift detection, and Redpanda does not perform RCA.

## Model bundle format

A bundle is a ZIP archive with this structure:

```text
my_model.drift.zip
├── model.pkl       # required: supported scikit-learn estimator or Pipeline
├── reference.csv  # required: baseline feature data
└── graph.json      # optional: directed dependency graph
```

- `model.pkl` is the trained model or scikit-learn Pipeline.
- `reference.csv` describes the reference input distribution.
- `graph.json` declares dependency nodes and edges used for graph-aware RCA.

A dependency graph is optional. Without it, model loading, predictions, and drift detection can still run, but dependency-based upstream tracing is unavailable. DriftTrace does not invent missing causal relationships.

> **Security:** Only upload trusted model bundles. Python pickle files can execute code when deserialized; do not load `model.pkl` files from untrusted sources.

### Generate sample bundles

The repository includes a script to generate five local validation bundles:

```powershell
python scripts/generate_test_models.py
python scripts/validate_test_models.py
```

Generated artifacts are placed under `artifacts/test_models/` and are runtime/test artifacts rather than source files. To locate the ZIP files on Windows:

```powershell
Get-ChildItem .\artifacts\test_models -Recurse -Filter *.drift.zip | Select-Object FullName
```

The **synthetic Gradient Boosting classifier with a dependency graph** is the recommended live demo model. It uses these feature relationships:

```text
engagement_score → service_score → risk_indicator → prediction
```

The generated bundles are not required for the application to start, but you need to upload and activate a model before using model-specific features. **These five bundles are regression fixtures, not an allowlist** — any model that conforms to a supported adapter contract works.

## Model-agnostic design

The monitoring core (drift engine, RCA engine, streaming lifecycle, frontend) does not depend on any specific model class, feature schema, feature count, class count, or output interface. It talks to models only through a **model adapter** (`src/drifttrace/adapters/base.py`) and selects one from bundle metadata via a registry (`src/drifttrace/adapters/registry.py`).

An adapter exposes framework/estimator identity, model/version/task metadata, the input feature names and order, prediction (and optional probability), the output shape, and clear errors for unsupported operations. The generic core only ever sees the standardized `PredictionResult` / `ModelMetadata` / `FeatureSchema` — never the framework object.

**Onboarding a previously unseen model** requires no production-code change:

1. Train any supported scikit-learn estimator or `Pipeline` (classifier, multiclass, or regressor; with or without `predict_proba`).
2. Save it as `model.pkl` (or `.joblib` / `.pickle`), export your baseline rows as `reference.csv`, and optionally declare a `graph.json`.
3. Zip them into a bundle and upload + activate it. The feature schema, task, output interface, and capabilities are discovered from the model and its reference data — DriftTrace uses the model's own feature names and order for prediction, monitoring, and explanations.

### Supported model/framework matrix

| Framework | Status | Notes |
| --- | --- | --- |
| scikit-learn estimators (classification, incl. multiclass) | **Implemented & tested** | Uses `feature_names_in_` / reference columns; `predict_proba` is optional |
| scikit-learn regressors | **Implemented & tested** | Numeric output; no class/probability required |
| scikit-learn `Pipeline` (preprocessing + estimator) | **Implemented & tested** | Preprocessing runs inside the model's own `predict` |
| XGBoost | **Not implemented** | No adapter yet; add one implementing the adapter contract |
| LightGBM | **Not implemented** | No adapter yet |
| ONNX Runtime | **Not implemented** | No adapter yet (would use `model.onnx`, not a pickle) |
| PyTorch / others | **Not implemented** | Future work |

A framework is listed as supported only when its adapter is implemented **and** independently tested. To add one, implement the `ModelAdapter` contract and register detection in `registry.py`; the drift engine, RCA engine, streaming lifecycle, and frontend do not need to change.

## Installation

### Requirements

- Git
- Python 3.11 or newer (Python 3.13.5 has been used successfully in the project environment)
- Node.js and npm (a current Node.js LTS release is recommended)
- A modern browser such as Chrome, Edge, Firefox, or Safari
- **Optional for Redpanda mode:** Docker Desktop with the Docker engine running

### 1. Clone the repository

```powershell
git clone https://github.com/shlokn678/DriftTrace.git
cd DriftTrace
```

### 2. Create and activate a Python virtual environment

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, follow your machine's existing PowerShell execution-policy guidance or activate the environment using your preferred terminal.

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

Run from the repository root with the virtual environment active:

```bash
python -m pip install --upgrade pip
pip install -e ".[serving,explain,dev,streaming]"
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 5. Generate and validate test bundles (optional, recommended)

```powershell
python scripts/generate_test_models.py
python scripts/validate_test_models.py
```

## Run locally without Redpanda

The broker-free local source is the easiest way to start DriftTrace. Docker and Redpanda are not required for this mode.

### Terminal 1 — backend

From the repository root, with `.venv` activated:

```powershell
$env:MLFLOW_DISABLE_AGENT_HINT="1"
python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`.

Health check: open `http://127.0.0.1:8000/health`. The expected response is similar to:

```json
{"status":"ok"}
```

### Terminal 2 — frontend

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173/`.

### Use the dashboard

1. Upload a `.drift.zip` model bundle.
2. Activate the model.
3. Use the batch/on-demand operator controls to run a drift test, inspect RCA, or explain a prediction.
4. For live monitoring in broker-free mode, start monitoring and use `POST /live/ingest` or the configured local event source to supply events.

## Run continuous monitoring with Redpanda

This section describes the **broker-backed live streaming path**. It is optional; the normal local application can run without a broker.

### Terminal 1 — start Docker Desktop and Redpanda

Start Docker Desktop and wait until its engine is ready. Verify Docker can connect to its engine:

```powershell
docker info --format '{{.ServerVersion}} / {{.OSType}}'
```

Then start Redpanda. Run this from PowerShell and keep the terminal open:

```powershell
docker run --rm --name drifttrace-redpanda -p 9092:9092 -p 9644:9644 `
  docker.redpanda.com/redpandadata/redpanda:v24.2.7 redpanda start `
  --overprovisioned --smp 1 --memory 1G --reserve-memory 0M --node-id 0 --check=false `
  --kafka-addr PLAINTEXT://0.0.0.0:9092 --advertise-kafka-addr PLAINTEXT://127.0.0.1:9092
```

Wait for Redpanda to report that startup succeeded and its Kafka API is listening on port `9092`. If a Redpanda container is already running, use that instance rather than starting another container with the same port mappings.

### Terminal 2 — start the backend in Redpanda mode

From the repository root, activate `.venv` and set the environment variables in the **same terminal** where you start the backend:

```powershell
.\.venv\Scripts\Activate.ps1

$env:MLFLOW_DISABLE_AGENT_HINT="1"
$env:DRIFTTRACE_LIVE_SOURCE="redpanda"
$env:DRIFTTRACE_USE_REDPANDA="true"
$env:REDPANDA_BROKER="127.0.0.1:9092"
$env:DRIFTTRACE_TOPIC="drifttrace.live.v4"
$env:DRIFTTRACE_LIVE_WINDOW_SIZE="200"
$env:DRIFTTRACE_LIVE_MIN_WINDOW_SAMPLES="50"

python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
```

The window size of `200` and minimum sample count of `50` are the recommended settings for the synthetic demo in this README. The implementation's window size is configurable through `DRIFTTRACE_LIVE_WINDOW_SIZE`.

### Terminal 3 — start the frontend

```powershell
cd frontend
npm run dev
```

Open the Vite URL in the browser.

### In the browser

1. Upload and activate `synthetic_gradient_boosting.drift.zip` from `artifacts/test_models/`. If you do not know the exact generated path, use the `Get-ChildItem` command in [Generate sample bundles](#generate-sample-bundles).
2. Open **Live Monitoring** and click **Start**.
3. Wait a few seconds for the live consumer to connect to the broker.

### Terminal 4 — send live demo events

Open a fourth terminal, navigate to the repository root, and activate the virtual environment:

```powershell
cd "C:\path\to\DriftTrace"
.\.venv\Scripts\Activate.ps1
```

Replace `C:\path\to\DriftTrace` with the actual project path. Run the following commands **one at a time**. Wait for the producer to finish and for the dashboard to process each set of events before starting the next stage.

**A. Normal traffic**

```powershell
python scripts/live_demo_producer.py --mode normal --events 600 --transport redpanda --brokers 127.0.0.1:9092 --topic drifttrace.live.v4
```

Expected: stable windows and no open incident, provided the generated normal observations match the reference distribution sufficiently closely.

**B. Inject drift**

```powershell
python scripts/live_demo_producer.py --mode drift --events 600 --transport redpanda --brokers 127.0.0.1:9092 --topic drifttrace.live.v4
```

Expected: drifted windows, per-feature KS/PSI values, root-cause diagnostics, and an incident after the configured number of consecutive drifted windows. For the showcase bundle, the expected chain is `engagement_score → service_score → risk_indicator`.

**C. Demonstrate recovery**

```powershell
python scripts/live_demo_producer.py --mode normal --events 1000 --transport redpanda --brokers 127.0.0.1:9092 --topic drifttrace.live.v4
```

Expected: stable windows return, and the incident closes after the configured consecutive-stable-window condition. The dashboard continues to show real KS/PSI measurements for the latest window; those measurements do not reset to zero, and prior windows remain in history.

### Running the producer for any other model

The producer is model-agnostic. It derives the feature schema and declared graph from the **active** model's reference data and `graph.json` — no feature names are hardcoded. Two options generalize it to any compatible bundle:

- `--bundle <path>`: derive the reference schema and declared graph from a specific bundle (`.zip` or directory). The bundle must match the active model's schema; a mismatch is rejected with an actionable error.
- `--drift-features f1,f2,...`: choose exactly which features DRIFT mode shifts, validated against the active model's real schema.

When neither is given, DRIFT defaults to the declared graph's upstream chain (so graph RCA yields one root + downstream symptoms), else the first numeric reference features. The producer prints which features it shifts and how the default was chosen.

```powershell
# a different, previously-unseen bundle (schema + graph derived from the bundle):
python scripts/live_demo_producer.py --bundle artifacts\test_models\wine_random_forest\wine_random_forest.drift.zip --mode normal --then-drift 300

# pick drift features explicitly (validated against the active model):
python scripts/live_demo_producer.py --mode drift --drift-features alcohol,proline
```

The existing synthetic showcase commands above are unchanged and remain the default.

### What to look for during the demo

- Normal traffic produces mostly **STABLE** windows.
- Drift traffic makes the drifted feature badges and KS/PSI diagnostics change.
- The live panel reports `engagement_score` as the root-cause candidate, with `service_score` and `risk_indicator` as downstream symptoms for the showcase graph.
- The dependency graph draws arrows only along edges declared in `graph.json`; unrelated stable features are not connected by invented edges.
- The incident opens after sustained drift and closes once enough stable windows have been evaluated.
- Model Explanation produces a local SHAP or LIME explanation, or a clear error with retry if a request fails or times out.

**Demo tip:** The live consumer processes events asynchronously. A producer can finish sending before all events have been evaluated. Give the dashboard time to catch up before judging the final count or incident state.

## API endpoints

The exact schemas are defined by the backend. The main live-monitoring endpoints are:

| Endpoint | Purpose |
|---|---|
| `POST /live/start` | Start monitoring for the active model; returns an error if no model is active. |
| `POST /live/stop` | Stop live monitoring. |
| `GET /live/status` | Return monitor status, counters, current-window progress, and the latest result. |
| `GET /live/windows?limit=N` | Retrieve persisted monitoring-window history. |
| `GET /live/latest` | Retrieve the latest completed live window. |
| `GET /live/incidents` | Retrieve incident state and incident history. |
| `POST /live/ingest` | Submit an event through the broker-free ingestion path. |

Other existing endpoints include `/predict`, `/demo/run-drift-test`, `/rca/latest`, `/models/*`, `/explain`, `/health`, and `/metrics`. Consult the route definitions and API response models for the current request/response schema.

## Multiple root-cause candidates

The RCA engine supports **multiple independent root-cause candidates in a single evaluated scenario**. The supplied synthetic showcase bundle contains one main dependency chain, so its normal drift demonstration identifies one root candidate.

Multi-root support is verified through a separate test-only graph with two independent chains:

```text
feature_a → feature_b

feature_c → feature_d
```

With genuine injected drift in both chains, the existing drift detector and RCA pipeline identified `feature_a` and `feature_c` as independent root-cause candidates, with `feature_b` and `feature_d` as their respective downstream symptoms. The live result schema preserves both candidates, and the frontend can display the returned root candidates. This does not imply that every incident has multiple roots; the result depends on the observed drift and the declared graph.

## Testing and validation

Run commands from the repository root with the project's virtual environment active.

### Backend tests

```powershell
python -m pytest -q
```

Focused multi-root RCA regression test:

```powershell
python -m pytest tests/integration/test_multi_root_rca.py -v
```

### Linting, formatting, and type checks

```powershell
python -m ruff check src tests
python -m black --check src tests
python -m mypy src
```

### Frontend checks

From the `frontend` directory:

```powershell
npx tsc --noEmit
npm run build
```

### Latest verified quality-gate results

The last recorded full validation for the implementation reported:

- **134 pytest tests passed**, 0 failures (includes genericity tests that build previously-unseen models at test time).
- Ruff: clean (`src tests`).
- Black: clean.
- MyPy: clean across 54 source files.
- Frontend TypeScript check: passed.
- Frontend production build: passed.
- Edge-faithful dependency-graph regression test: passed.
- Multi-root RCA tests with genuine drift data and live result-schema verification: passed.
- Model-agnostic end-to-end tests (`tests/integration/test_model_agnostic.py`): previously-unseen multiclass `Pipeline`, a regressor without `predict_proba` (SVR), a novel graph chain with edge-accurate RCA, model-switch isolation, and dynamic-schema SHAP — all pass with **no production-code changes per model**.
- Smoke test and five-bundle validation: passed.

These are recorded validation results for the tested code state. After making further code changes, rerun the commands above to validate the new state.

## Project structure

```text
DriftTrace/
├── src/drifttrace/             # backend and core monitoring logic
│   ├── graph/                  # dependency graph implementation
│   ├── governance/             # report construction and RCA/report integration
│   ├── serving/                # FastAPI endpoints and serving configuration
│   └── streaming/              # event schema, live monitor, persistence
├── frontend/                   # React + TypeScript + Vite dashboard
├── tests/                      # backend unit and integration tests
├── scripts/                    # demo producer, validation, and test-model tools
├── artifacts/                  # generated local model bundles (runtime artifacts)
├── reports/                    # generated reports and live-monitoring history
├── config/                     # runtime configuration
├── pyproject.toml              # package metadata and dependencies
└── README.md
```

## Concepts

| Term | Meaning in DriftTrace |
|---|---|
| **Reference data / baseline** | Data representing the distribution the active model is expected to receive. |
| **Data drift** | A change in the distribution of incoming data relative to the reference. Drift alone does not prove model accuracy has fallen. |
| **KS statistic** | A measure of the maximum separation between two empirical cumulative distributions. |
| **KS p-value** | Statistical evidence used with the KS test; it is not an effect-size measure by itself. |
| **PSI** | A measure of population shift between reference and current data. |
| **Monitoring window** | A group of events evaluated together. Live monitoring uses configurable tumbling windows. |
| **Dependency graph** | A directed graph of declared relationships used for tracing affected nodes. |
| **Root-cause candidate** | An upstream affected node supported by the available drift results and dependency graph; not proof of causality. |
| **Symptom** | A downstream affected node on a dependency path. |
| **SHAP/LIME** | Methods for explaining feature influence on an individual prediction, distinct from drift RCA. |
| **Human-in-the-loop** | Operators remain responsible for approving significant operational actions. |

## Limitations and future work

1. **Model adapters:** The implemented model adapter supports scikit-learn estimators and Pipelines. Support for other frameworks such as XGBoost, LightGBM, ONNX, or PyTorch is future work unless explicitly added and validated.
2. **Graph quality:** Root-cause tracing depends on the supplied dependency graph. DriftTrace uses declared edges and measured drift; it does not prove causal relationships or automatically propagate drift through the graph.
3. **Reference baseline:** Drift measurements are relative to the active model's reference data. DriftTrace does not automatically update the baseline.
4. **Statistical thresholds:** PSI/KS thresholds need to be chosen for the problem, sample size, and acceptable false-positive rate. Small windows can yield noisy PSI values.
5. **Production deployment:** The project is local-first. Cloud deployment, authentication/authorization hardening, production-grade broker operations, external alert integrations, and load testing are future work.
6. **External monitoring stack:** Prometheus/Grafana integration is not part of the current core dashboard implementation.

Possible extensions include sliding windows, external alert destinations, broader model adapters, cloud deployment, automatic graph learning, temporal dependency reasoning, and graph-neural-network approaches.

## Project goal in one line

> DriftTrace turns “the model is drifting” into a more actionable question: **“Which related part changed, what evidence supports that diagnosis, and what should the human operator investigate next?”**
