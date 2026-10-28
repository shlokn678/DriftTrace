# DriftTrace

**An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift**

DriftTrace monitors a deployed machine-learning model, detects distribution changes in incoming data, and uses a declared dependency graph to help identify likely root-cause candidates and downstream symptoms. It supports batch/on-demand monitoring as well as continuous event-stream monitoring, with a React dashboard for diagnostics and incident history.

> **Detect drift → Trace dependencies → Identify likely root causes → Show symptoms → Let a human decide what to do**

DriftTrace is model-agnostic across its supported **scikit-learn estimators and Pipelines**. A fresh checkout does not include a built-in production model: you upload a model bundle or generate the local test bundles.

## Contents

- [Project overview](#project-overview)
- [Features](#features)
- [Architecture and data flow](#architecture-and-data-flow)
- [Technology stack](#technology-stack)
- [Model bundle format](#model-bundle-format)
- [Model-agnostic design](#model-agnostic-design)
- [Installation](#installation)
- [Run locally without Redpanda](#run-locally-without-redpanda)
- [Run continuous monitoring with Redpanda](#run-continuous-monitoring-with-redpanda)
- [Explainability: SHAP and LIME](#explainability-shap-and-lime)
- [API endpoints](#api-endpoints)
- [Multiple root-cause candidates](#multiple-root-cause-candidates)
- [Testing and validation](#testing-and-validation)
- [Project structure](#project-structure)
- [MLflow experiment tracking](#mlflow-experiment-tracking)
- [DVC data versioning and validation](#dvc-data-versioning-and-validation)
- [Limitations and future work](#limitations-and-future-work)
- [Troubleshooting](#troubleshooting)
- [Glossary of technical terms](#glossary-of-technical-terms)
- [Conclusion](#conclusion)

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

- **KS test (continuous features):** reports the KS statistic and p-value for numerical feature distributions.
- **PSI:** measures the shift between current and reference populations, for both continuous and categorical distributions.
- **Categorical drift:** compares category frequencies where applicable.
- **Prediction/output drift:** monitors the model's output distribution against a reference output baseline. For a **classifier**, the output is a discrete class label, so it is monitored with **categorical class-frequency PSI** — the KS statistic and KS p-value are legitimately shown as unavailable (`-`) for a categorical output while a valid PSI is still reported. For a **regressor**, the numeric output uses KS + PSI like a continuous feature. Output observations are collected from the scored prediction events, so the output drift signal updates during live monitoring.

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

The explanation form is **pre-filled with one representative row** drawn from the active model's reference data (real, valid values in the model's own feature order), so the panel does not present an all-zeros example. The inputs stay editable, and if no reference row is available the form stays blank and prompts the user to enter values. Explanation requests have cancellation, timeout handling, and a retry/error state. Feature contribution describes the direction and relative magnitude of a feature's influence on the explained prediction; it is **not** the same as a causal root cause and is not a percentage. See [Explainability: SHAP and LIME](#explainability-shap-and-lime).

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

## Technology stack

Versions below are the lower bounds declared in `pyproject.toml` (backend) and `frontend/package.json` (frontend), verified against the repository. Actual installed versions may be newer.

| Technology | Version (min / pinned) | Role in DriftTrace |
| --- | --- | --- |
| Python | >= 3.11 | Backend language (3.13 used in the project environment). |
| FastAPI | >= 0.110 | HTTP API for model onboarding, prediction, drift tests, live monitoring, and explanations. |
| Uvicorn | >= 0.29 | ASGI server that runs the FastAPI app. |
| scikit-learn | >= 1.3 | The one implemented model family (estimators and Pipelines). |
| NumPy / pandas | >= 1.26 / >= 2.1 | Numerical arrays and reference/window data handling. |
| SciPy | >= 1.11 | KS two-sample test used by the drift engine. |
| NetworkX | >= 3.2 | Represents and traverses the declared dependency graph for RCA. |
| Pydantic | >= 2.5 | Request/response validation and the standardized event/model schemas. |
| SHAP | >= 0.44 | Local feature-attribution explanations for a single prediction. |
| LIME | >= 0.2.0.1 | Alternative local explanation method. |
| confluent-kafka | >= 2.3 | Kafka-compatible client used to produce/consume events via Redpanda. |
| React | ^18.3 | Dashboard UI library. |
| TypeScript | ^5.6 | Typed JavaScript for the frontend. |
| Vite | ^5.4 | Frontend dev server and production build tool. |
| lucide-react | ^0.454 | Icon set used in the dashboard. |
| Redpanda | image `v24.2.7` | Kafka-API-compatible event broker for the live streaming demo (run via Docker). |
| MLflow | >= 2.11 (3.x installed) | Supporting experiment tracking for the five test models' pre-recorded metrics. |
| DVC | installed separately (3.x) | Data versioning + a one-stage validation pipeline for the demo data. Not declared in `pyproject.toml`; install it into the environment to use the DVC section. |
| pytest / Ruff / Black / mypy | >= 8.0 / >= 0.4 / >= 24.3 / >= 1.9 | Testing, linting, formatting, and static type checking (the `dev` extra). |

Install extras are defined in `pyproject.toml`: `serving` (FastAPI/Uvicorn/Kafka client), `explain` (SHAP/LIME), `streaming`, `tracking` (MLflow), and `dev` (test/quality tools).

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

This repository does **not** contain a root-level Docker Compose file. Do not run `docker compose`; use the `docker` container commands below. Docker is only used to run the Redpanda broker locally — the DriftTrace application itself is not containerized.

The verified setup uses a **persistent, named** Redpanda container called `drifttrace-redpanda-1` (image `redpandadata/redpanda:v24.2.7`) that advertises its Kafka API at `127.0.0.1:9092`.

Start Docker Desktop first and wait until its engine is ready. Verify Docker can connect to its engine:

```powershell
docker info --format '{{.ServerVersion}} / {{.OSType}}'
```

#### If the container already exists (the common case)

Start the existing container and keep its data; never delete your existing data volume or recreate the container just to start it:

```powershell
docker start drifttrace-redpanda-1
```

Verify the broker is healthy and reachable:

```powershell
docker exec drifttrace-redpanda-1 rpk cluster info
Test-NetConnection 127.0.0.1 -Port 9092
```

`rpk cluster info` should list one broker; `Test-NetConnection` should report `TcpTestSucceeded : True`. Leave the broker running for the rest of the demo.

#### If no Redpanda container exists yet (new machine)

Provision one **once**, with a named persistent volume so monitoring history survives restarts, using the tested image and advertising `127.0.0.1:9092` (which is what the producer on Windows connects to):

```powershell
docker volume create drifttrace-redpanda-data

docker run -d --name drifttrace-redpanda-1 `
  -p 9092:9092 -p 9644:9644 `
  -v drifttrace-redpanda-data:/var/lib/redpanda/data `
  redpandadata/redpanda:v24.2.7 redpanda start `
  --overprovisioned --smp 1 --memory 1G --reserve-memory 0M --node-id 0 --check=false `
  --kafka-addr PLAINTEXT://0.0.0.0:9092 --advertise-kafka-addr PLAINTEXT://127.0.0.1:9092
```

This creates a **persistent** container (`-d`, no `--rm`) backed by the `drifttrace-redpanda-data` volume. After it exists, you start and stop it with `docker start drifttrace-redpanda-1` / `docker stop drifttrace-redpanda-1` — do not run the `docker run` command again, and do not delete the volume, or you will lose broker data. Then verify it with the `rpk cluster info` and `Test-NetConnection` checks above.

> **Advertised address matters.** The broker must advertise `127.0.0.1:9092`. An earlier setup advertised `redpanda:9092` (a Docker-internal hostname), which the producer running on the Windows host could not resolve, so events never reached the broker. The commands above advertise `127.0.0.1:9092`, which works from the Windows host. See [Troubleshooting the live demo](#troubleshooting-the-live-demo) if events do not arrive.

### Terminal 2 — start the backend in Redpanda mode

Set the environment variables in the **same terminal** where you start the backend; they must be present in that shell before `serve` runs. Copy the whole block:

```powershell
cd "C:\Users\shlok\OneDrive\Documents\College (sem 5)\MLOps\CP"
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

Replace the `cd` path if your repository lives elsewhere. Wait until the backend logs `Application startup complete` (Uvicorn), then leave this terminal running. The window size of `200` and minimum sample count of `50` are the recommended settings for the synthetic demo; both are configurable through `DRIFTTRACE_LIVE_WINDOW_SIZE` and `DRIFTTRACE_LIVE_MIN_WINDOW_SAMPLES`.

If you recreate the Redpanda container after the backend has started, restart the backend (stop with `Ctrl+C`, then re-run the block above) so it reconnects to the new broker.

### Terminal 3 — start the frontend

```powershell
cd "C:\Users\shlok\OneDrive\Documents\College (sem 5)\MLOps\CP\frontend"
npm run dev
```

Open the Vite URL printed in this terminal, usually `http://localhost:5173/`.

### In the browser

1. Upload and activate the synthetic Gradient Boosting bundle, `synthetic_gradient_boosting.drift.zip`, from `artifacts/test_models/`. If you do not know the exact generated path, use the `Get-ChildItem` command in [Generate sample bundles](#generate-sample-bundles).
2. Confirm the dashboard shows the intended demo model as **active** and that the live source is **redpanda** (shown in the Live Monitoring panel / `GET /live/status`), not the broker-free file source.
3. Open **Live Monitoring** and click **Start**.
4. Wait a few seconds for the live consumer to connect to the broker.

### Terminal 4 — send live demo events

Open a fourth terminal, go to the repository root, and activate the virtual environment:

```powershell
cd "C:\Users\shlok\OneDrive\Documents\College (sem 5)\MLOps\CP"
.\.venv\Scripts\Activate.ps1
```

Replace the path if your repository lives elsewhere. Run the three commands **one at a time, in order**. After each one finishes, check the dashboard and confirm the events and evaluated windows before starting the next phase. The producer's own progress messages confirm only that it sent events — they do not prove the broker delivered them or that the backend evaluated a window. Always confirm on the dashboard (or `GET /live/status` / `GET /live/windows`).

**A. Normal baseline**

```powershell
python scripts/live_demo_producer.py --mode normal --events 600 --transport redpanda --brokers 127.0.0.1:9092 --topic drifttrace.live.v4
```

Expected flow: mostly stable windows and no open incident, provided the generated normal observations track the reference distribution closely enough.

**B. Inject drift**

```powershell
python scripts/live_demo_producer.py --mode drift --events 600 --transport redpanda --brokers 127.0.0.1:9092 --topic drifttrace.live.v4
```

Expected flow: drifted windows with per-feature KS/PSI values and graph-guided root-cause diagnostics, and a drift incident once enough consecutive drifted windows are evaluated. For the showcase bundle the declared chain is `engagement_score → service_score → risk_indicator`, so `engagement_score` is reported as the root-cause candidate with `service_score` and `risk_indicator` as downstream symptoms.

**C. Recovery**

```powershell
python scripts/live_demo_producer.py --mode normal --events 1000 --transport redpanda --brokers 127.0.0.1:9092 --topic drifttrace.live.v4
```

Expected flow: subsequent windows return to stable/warning as normal data resumes, and the incident closes once enough consecutive stable windows are evaluated. The dashboard keeps showing real KS/PSI measurements for the latest window, and prior windows remain in history.

> **Results are not fixed numbers.** Exact event totals, per-window statuses, the window at which an incident opens, and the window at which it closes depend on the current monitoring state and the observations actually evaluated. Treat the flow above as the qualitative sequence (normal → drift + RCA incident → recovery), not as guaranteed counts.

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

### Troubleshooting the live demo

All checks below are non-destructive — they inspect state and never delete a container, volume, or topic.

- **Container health:** confirm the broker is up and lists a broker node.

  ```powershell
  docker ps --filter "name=drifttrace-redpanda-1"
  docker exec drifttrace-redpanda-1 rpk cluster info
  ```

- **Advertised broker address:** the broker must advertise `127.0.0.1:9092`. If an older container advertises `redpanda:9092`, the Windows-host producer cannot resolve it and events never arrive. Inspect the advertised address:

  ```powershell
  docker exec drifttrace-redpanda-1 rpk cluster info
  ```

  The reported broker address should be `127.0.0.1:9092`. If it is `redpanda:9092` (or another internal hostname), re-provision the container using the new-machine command in [Run continuous monitoring with Redpanda](#run-continuous-monitoring-with-redpanda) (advertising `127.0.0.1:9092`) rather than editing the running container. Do not delete the data volume as part of this.

- **Topic:** confirm the demo topic exists and has received records (it is created on first produce).

  ```powershell
  docker exec drifttrace-redpanda-1 rpk topic list
  docker exec drifttrace-redpanda-1 rpk topic describe drifttrace.live.v4
  ```

- **Port 9092 from the host:**

  ```powershell
  Test-NetConnection 127.0.0.1 -Port 9092
  ```

  Expect `TcpTestSucceeded : True`.

- **Backend status:**

  ```powershell
  (Invoke-WebRequest http://127.0.0.1:8000/health).Content
  (Invoke-WebRequest http://127.0.0.1:8000/live/status).Content
  ```

  `/live/status` should report the active model, `source` as `redpanda`, and increasing event counters while the producer runs.

- **Dashboard event counts:** watch the Live Monitoring panel's events-received and windows-evaluated counters, or poll `GET /live/status` and `GET /live/windows`. Producer progress messages alone do not prove delivery — only the dashboard/API counters confirm that events reached the broker and that windows were evaluated.

- **Broker recreated after the backend started:** if you stop/recreate the Redpanda container while the backend is running, restart the backend (`Ctrl+C`, then re-run the [Terminal 2](#terminal-2--start-the-backend-in-redpanda-mode) block) so it reconnects to the new broker.

## Explainability: SHAP and LIME

The dashboard's **Model Explanation** panel explains a single prediction for the active model, separately from live drift monitoring.

How it works:

1. The panel reads the active model's feature schema and a **representative reference row** from `GET /model-info` (`sample_row`), and pre-fills one input per feature with those real values in the model's feature order. If no reference row is available, the inputs stay blank and the panel asks you to enter values rather than defaulting to zeros.
2. You can edit any input. Clicking **Explain prediction** sends the current input values to `POST /explain`, which scores the active model and computes a SHAP (default) or LIME explanation for that prediction.
3. The panel shows the predicted class or value, a short natural-language summary, the supporting and opposing feature factors, and (behind a toggle) the raw attribution values.

Interpreting the values honestly:

- A **positive** attribution means the feature pushed the prediction **toward** the explained class/output; a **negative** attribution means it pushed **away**. The magnitude indicates relative strength, **not a percentage**.
- Feature attribution explains *this model's* behaviour on *this input*. It is **not** proof of a real-world causal relationship, and it is distinct from dependency-graph RCA.
- Explanations are computed on demand for one prediction. They are **not** produced automatically for every live monitoring window; live drift monitoring and prediction explanations are separate features.

Explanation requests run off the prediction hot path and include request cancellation, a timeout, and a visible retry/error state so a slow or failed request does not leave the panel stuck.

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

Other existing endpoints include `/health`, `/ready`, `/model-info` (feature schema plus a representative `sample_row`), `/predict`, `/explain`, `/demo/run-drift-test`, `/rca/latest`, `/metrics`, and the model-management routes `/models/upload`, `/models/active`, `/models/{model_id}`, `/models/{model_id}/activate`, and `/models/deactivate`. Consult the route definitions and API response models for the current request/response schema.

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

### Model-bundle and DVC validation

```powershell
# Regenerate (optional) and validate the five demonstration bundles end to end.
python scripts/generate_test_models.py
python scripts/validate_test_models.py

# Validate the DVC-tracked demo data (same check wired as the DVC stage).
python -m dvc repro
python -m dvc metrics show
```

### Latest recorded quality-gate results

The following are the **last recorded** verification results for the tested code state. They were observed in a previous verification run and are reported here as recorded results — rerun the commands above to re-confirm them on your machine.

- **145 backend pytest tests passed**, 0 failures (the suite currently collects 145 tests; it includes genericity tests that build previously-unseen models at test time, categorical prediction-output PSI tests, and the SHAP reference-row tests).
- Ruff: passed (`src tests`).
- Black: passed.
- mypy: passed.
- Frontend TypeScript check (`npx tsc --noEmit`): passed.
- Frontend production build (`npm run build`): passed.
- Five-bundle validation (`scripts/validate_test_models.py`): all five models passed.
- DVC validation stage: `mlops_demo_summary.json` reports `all_valid: true` for all five datasets.

## Project structure

```text
DriftTrace/
├── src/drifttrace/             # backend and core monitoring logic
│   ├── adapters/               # model adapter interface + scikit-learn adapter + registry
│   ├── bundle/                 # bundle loading, reference profiling, baseline building
│   ├── drift/                  # KS/PSI drift engine and verdicts
│   ├── rca/                    # dependency-graph root-cause analysis
│   ├── graph/                  # dependency graph representation and traversal
│   ├── explain/                # SHAP / LIME local explanations
│   ├── streaming/              # event schema, live monitor, windows, persistence
│   ├── serving/                # FastAPI endpoints, onboarding, serving configuration
│   ├── alerting/               # incident alert payloads and delivery helpers
│   ├── governance/             # monitoring report construction / RCA-report integration
│   └── cli/                    # command-line entry point (serve, etc.)
├── frontend/                   # React + TypeScript + Vite dashboard
│   └── src/                    # api client/types, sections, components, hooks
├── tests/                      # backend unit and integration tests (pytest)
├── scripts/                    # demo producer, model-bundle tools, MLflow/DVC utilities
│   ├── live_demo_producer.py   # Redpanda/broker-free event producer for the live demo
│   ├── generate_test_models.py # builds the five demonstration bundles
│   ├── validate_test_models.py # end-to-end validation of the five bundles
│   ├── log_test_models_mlflow.py     # MLflow: logs pre-recorded metrics as runs
│   └── validate_mlops_demo_data.py   # DVC stage: validates the demo data
├── config/                     # runtime configuration (drift/window defaults)
├── artifacts/test_models/      # generated local model bundles (runtime artifacts)
├── reports/                    # generated reports and live-monitoring history
├── mlops_demo_data/            # DVC-tracked copy of the demo datasets
├── mlops_demo_data.dvc         # DVC pointer file for mlops_demo_data/
├── dvc.yaml / dvc.lock         # DVC pipeline (validate_demo_data stage) + lock file
├── mlops_demo_summary.json     # DVC metrics output from the validation stage
├── mlflow.db                   # local MLflow SQLite tracking store (generated)
├── pyproject.toml              # package metadata, dependencies, and tool config
└── README.md
```

The tree omits incidental, machine-specific items that are not core source — for example the Python virtual environment (`.venv/`), `node_modules/`, build output, caches, and the generated MLflow/DVC cache directories.

## MLflow experiment tracking

MLflow is used here as **supporting experiment tracking** for the five demonstration models. It is deliberately limited and is **not** part of the live drift-detection / RCA engine. It does **not** perform automatic retraining, automatic model selection, or any automated production training pipeline.

What the logger does: `scripts/log_test_models_mlflow.py` reads each model's existing `metrics.json` (written when the bundles were generated) and records one MLflow run per model — parameters (model name, algorithm, task, dataset, feature count, graph availability) and the pre-recorded scalar metrics — under the experiment **`DriftTrace-Test-Models`**. The tracking store is a local SQLite file, `mlflow.db`, at the repository root. Each run is tagged (`metrics_source = precomputed_from_metrics_json`) to make clear that **nothing is retrained and no metric is re-evaluated**. The source `metrics.json` is attached to each run as an artifact.

Run it from the repository root with the virtual environment active (requires the `tracking` extra: `pip install -e ".[tracking]"`):

```powershell
.\.venv\Scripts\Activate.ps1

# Log the five runs into mlflow.db (one run per model).
python scripts/log_test_models_mlflow.py
```

> **Note:** running the logger again creates **additional** runs — it appends, it does not overwrite. Re-run it only when you want new copies of the runs.

Open the MLflow UI against the same local store (this is a long-running server; stop it with `Ctrl+C`):

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Then browse to `http://127.0.0.1:5001`, open the `DriftTrace-Test-Models` experiment, and compare the five runs. The port is whatever you pass to `--port`; `5001` is used here to avoid clashing with other local services. The logged metrics are the held-out evaluation numbers recorded for each model (for example, the `synthetic_regression` run shows `r2 ≈ 0.956`); they come straight from the bundles' `metrics.json`, not from any run performed by DriftTrace monitoring.

## DVC data versioning and validation

DVC is used here for **data versioning and a small validation pipeline** over the demonstration data. Like MLflow, it is a supporting utility separate from the live monitoring engine.

What is tracked and validated:

- `mlops_demo_data/` holds a copy of each model's `reference.csv`, `stable.csv`, `drift.csv`, and `metrics.json`. The directory is tracked by DVC via `mlops_demo_data.dvc` (created with `dvc add`), so the data content is versioned by hash rather than committed directly to Git.
- `dvc.yaml` defines a single pipeline stage, **`validate_demo_data`**, which runs `scripts/validate_mlops_demo_data.py`. The script checks, for each of the five models, that `reference.csv`, `stable.csv`, `drift.csv`, and `metrics.json` all exist and that the three CSVs share the same feature columns, then records row counts, feature count, task, algorithm, and the scalar metrics.
- The stage writes `mlops_demo_summary.json` (the DVC metrics output). `dvc.lock` records the resolved stage inputs/outputs.

**Storage (be explicit):** a DVC remote named `localremote` is configured to a **local filesystem path** (`../.dvc-storage`) in `.dvc/config`. This demonstrates local reproducibility and `dvc push`/`dvc pull` against on-disk storage, but it is **not** a shared or cloud remote — the data is not automatically backed up to or shared with another computer unless you configure and populate a shared remote. DVC must be installed in the environment (it is not declared in `pyproject.toml`): `pip install dvc`.

Run the pipeline from the repository root with the virtual environment active:

```powershell
.\.venv\Scripts\Activate.ps1

# Reproduce the validation stage (writes mlops_demo_summary.json).
python -m dvc repro

# Check whether the tracked data / stage outputs are up to date.
python -m dvc status

# Show the metrics the stage produced.
python -m dvc metrics show
```

Expected result when the current artifacts are valid: all five demonstration datasets validate successfully — `mlops_demo_summary.json` reports `"all_valid": true` with `"n_models": 5` and every model's `"status": "pass"` (four classification datasets and one regression dataset). If `dvc status` reports the stage is already up to date, `dvc repro` will not re-run it; use `dvc repro --force` only if you intend to re-execute the validation.

## Limitations and future work

1. **Model adapters:** The implemented model adapter supports scikit-learn estimators and Pipelines. Support for other frameworks such as XGBoost, LightGBM, ONNX, or PyTorch is future work unless explicitly added and validated.
2. **Graph quality:** Root-cause tracing depends on the supplied dependency graph. DriftTrace uses declared edges and measured drift; it does not prove causal relationships or automatically propagate drift through the graph.
3. **Reference baseline:** Drift measurements are relative to the active model's reference data. DriftTrace does not automatically update the baseline.
4. **Statistical thresholds:** PSI/KS thresholds need to be chosen for the problem, sample size, and acceptable false-positive rate. Small windows can yield noisy PSI values.
5. **Production deployment:** The project is local-first. Cloud deployment, authentication/authorization hardening, production-grade broker operations, external alert integrations, and load testing are future work.
6. **External monitoring stack:** Prometheus/Grafana integration is not part of the current core dashboard implementation.
7. **MLOps tooling is a limited, supporting demonstration only:** MLflow and DVC are wired in as small, standalone utilities, not as part of the live drift-detection / RCA engine. MLflow is used for experiment tracking of the five test models' **pre-recorded** metrics (read from each bundle's `metrics.json`; nothing is retrained or re-evaluated). DVC versions a copy of the demo data (`mlops_demo_data/`) and runs a single validation stage that produces a metrics report. The DVC remote is **local-only** (no cloud storage). See the "MLflow and DVC demonstration" section for how to run them. Apache Airflow remains **not** integrated. Automatic graph learning and graph-neural-network (GNN) based RCA are also not implemented; RCA is graph-traversal over a declared dependency graph, not a learned model. Treat these as future work.
8. **Containerization:** Docker is used only to run the Redpanda broker locally. The DriftTrace backend and frontend are not containerized, and there is no root-level Docker Compose file.

Possible extensions include sliding windows, external alert destinations, broader model adapters, cloud deployment (including managed services such as SageMaker), automatic graph learning, temporal dependency reasoning, and graph-neural-network approaches.

## Troubleshooting

Non-destructive checks for the most common problems. None of these steps delete data, model artifacts, DVC storage, or Docker volumes. (For live-streaming-specific issues, see also [Troubleshooting the live demo](#troubleshooting-the-live-demo).)

| Symptom | Likely cause | Safe check / action |
| --- | --- | --- |
| `/predict`, `/explain`, or live start returns an error about no active model | No model has been uploaded and activated in this session | Upload a `.drift.zip` bundle and activate it; confirm with `GET /models/active`. |
| Dashboard shows a model but drift/RCA panels stay empty | No monitoring window has completed yet | Run a drift test, or send enough live events to fill at least one window (`DRIFTTRACE_LIVE_WINDOW_SIZE`); check `GET /live/windows`. |
| `prediction (output)` card shows KS statistic / KS p-value as `-` with a PSI value present | The active model is a classifier, so its output is categorical | This is expected: categorical output uses class-frequency PSI, not the KS test. A present PSI with blank KS is correct, not an error. |
| A drift test or live window reports insufficient data | Window has fewer samples than `DRIFTTRACE_LIVE_MIN_WINDOW_SAMPLES` | Send more events or lower the minimum-sample setting for the demo. |
| Producer prints progress but the dashboard counters do not move | Events are not reaching the broker/backend (e.g. wrong broker address or topic) | Confirm the broker advertises `127.0.0.1:9092`, the topic matches `DRIFTTRACE_TOPIC`, and `GET /live/status` shows `source: redpanda`; see the live-demo troubleshooting section. |
| Backend cannot connect to Redpanda | Broker not started, or backend started before the broker | Start/verify the container (`docker start drifttrace-redpanda-1`, `rpk cluster info`), then restart the backend so it reconnects. |
| Uploading a bundle fails validation | Missing `model.pkl` / `reference.csv`, or an unsupported model type | Check the bundle contains the required files and a supported scikit-learn model; read the error message returned by `/models/upload`. |
| Producer rejects a `--bundle` with a schema-mismatch error | The bundle's feature schema differs from the active model's | Activate the matching model first, or pass a bundle whose `reference.csv` columns match the active model. |
| Live panel seems to show a previous model's results | A stale session before switching models | Monitoring state is scoped by model/session; re-check `GET /live/status`, and start a new monitoring session for the current model. |
| `mlflow ui` port is already in use | Another process holds the port | Choose a different `--port` (for example `5002`); the UI port is not fixed by the app. |
| `dvc repro` says "up to date" and does nothing | Inputs and outputs are unchanged | This is normal. Use `python -m dvc repro --force` only if you intend to re-run the validation stage. |

## Glossary of technical terms

Plain-English definitions (one to two sentences each) for the terms used in this README and the project.

| Term | Simple explanation |
| --- | --- |
| MLOps | The practice of running machine-learning models reliably in production, including deployment, monitoring, and maintenance. |
| Machine learning model | A program that learns patterns from data and then makes predictions on new data. |
| Model bundle | A ZIP file containing a trained model, its reference data, and an optional dependency graph that DriftTrace loads and monitors. |
| Reference data / baseline | A saved sample of the data the model is expected to see, used as the "normal" to compare new data against. |
| Feature | One input column the model uses to make a prediction (for example, "age" or "price"). |
| Prediction / model output | The value the model produces for an input: a class label for classification or a number for regression. |
| Data drift | A change over time in the distribution of the input data compared with the reference data. |
| Concept drift | A change in the real relationship between inputs and the correct answer, which data drift alone does not directly measure. |
| Continuous feature | A numeric feature that can take a range of values (for example, temperature). |
| Categorical feature | A feature whose values are distinct labels or categories (for example, "red" / "blue"). |
| Monitoring window | A fixed-size group of recent events that DriftTrace evaluates together as one unit. |
| KS test | The Kolmogorov–Smirnov test, a statistical test that compares two numeric distributions to see how different they are. |
| KS statistic | A number from the KS test giving the largest gap between the two distributions; larger means more different. |
| P-value | The probability of seeing a difference this large by chance if nothing really changed; a small value suggests a real difference. |
| Population Stability Index (PSI) | A score that measures how much a distribution has shifted from the reference; higher means a bigger shift. |
| Drift threshold | A configured cut-off (for example PSI ≥ 0.20) above which a change is treated as drift. |
| Drift detection | The process of deciding whether current data differs enough from the reference to be flagged as drifted. |
| Root-cause analysis (RCA) | Using the dependency graph and drift results to suggest which upstream feature most likely started an observed drift. |
| Dependency graph | A map of declared directed relationships between features, showing which feature is considered upstream of another. |
| Root node | In RCA, an upstream feature flagged as a likely starting point of the drift (a candidate, not proven cause). |
| Symptom / downstream feature | A feature further down the dependency graph whose drift is likely a knock-on effect of an upstream change. |
| Incident | A tracked event opened when drift persists across consecutive windows, so repeated drift is grouped instead of re-alerted each window. |
| Recovery | The return to stable windows after drift, which eventually closes an open incident. |
| SHAP | A method that estimates how much each feature contributed to one specific prediction. |
| LIME | Another method that explains one prediction by approximating the model locally around that input. |
| Feature attribution | A value showing the direction and relative strength of a feature's influence on a prediction; it is not a percentage or a proven cause. |
| Causal relationship | A real cause-and-effect link; DriftTrace does not prove this, it only provides diagnostic evidence. |
| Model adapter | The small interface that lets the monitoring core talk to a model without depending on its specific framework. |
| Classification | A prediction task whose output is a category or class label. |
| Regression | A prediction task whose output is a continuous number. |
| API | An application programming interface: the set of HTTP endpoints other programs use to talk to the backend. |
| FastAPI | The Python web framework used to build DriftTrace's HTTP API. |
| React | The JavaScript library used to build the dashboard user interface. |
| TypeScript | A typed version of JavaScript used for the frontend code. |
| Vite | The tool that runs the frontend in development and builds it for production. |
| Streaming | Continuously sending and processing data events one after another, rather than in a single batch. |
| Event producer | A program (here `scripts/live_demo_producer.py`) that sends prediction events into the broker for monitoring. |
| Redpanda | A Kafka-compatible event broker, run locally in Docker, that carries the streamed prediction events. |
| Kafka-compatible broker | A message system that speaks the Kafka protocol, so Kafka clients can connect to it (Redpanda is one). |
| MLflow | A tool for tracking machine-learning experiments; here it records the five test models' pre-computed metrics. |
| Experiment tracking | Recording models, parameters, and metrics so runs can be compared later. |
| Metric | A measured number describing model quality, such as accuracy or R². |
| DVC | Data Version Control, a tool for versioning datasets and defining reproducible data pipelines. |
| Data versioning | Tracking changes to datasets by content so a specific version can be restored or reproduced. |
| Reproducibility | Being able to re-run a process and get the same result from the same inputs. |
| DVC pipeline / stage | A defined step (here `validate_demo_data`) with declared inputs and outputs that DVC can re-run on demand. |
| pytest | The framework used to run the backend's automated tests. |
| Ruff | A fast Python linter that flags code-style and quality issues. |
| Black | A Python code formatter that enforces a consistent style. |
| mypy | A static type checker that verifies Python type annotations without running the code. |
| Tumbling window | A monitoring window that does not overlap with the next one; each event belongs to exactly one window. |
| Live source | Where live events come from: the broker-free local path, or a Redpanda topic. |

## Conclusion

DriftTrace is a local-first MLOps monitoring and investigation tool for scikit-learn models. You upload a model bundle (model, reference data, optional dependency graph), then monitor it either on demand or as a continuous Redpanda event stream. The same KS/PSI drift engine and dependency-graph RCA engine run in both paths: DriftTrace detects feature and prediction-output drift, groups persistent drift into incidents that open and recover, traces likely root-cause candidates when a graph is provided, and explains individual predictions with SHAP or LIME. MLflow and DVC are included as supporting experiment-tracking and data-validation utilities, not as an automated training or production pipeline.

It is intended for learning, demonstration, and local experimentation. It is not a hardened production deployment: there is no authentication layer, no automatic retraining or rollback, and storage/broker operations are local. Used within those limits, DriftTrace turns “the model is drifting” into a more actionable question — **which related part changed, what evidence supports that diagnosis, and what should the human operator investigate next?**
