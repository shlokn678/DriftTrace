# DriftTrace

**An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift**

DriftTrace is a student MLOps project that monitors a machine-learning model after deployment, detects when the input data has changed, and helps identify **where that change most likely started**.

The main idea is simple:

> **Detect drift → Trace dependencies → Identify a likely root cause → Show symptoms → Let a human decide what to do**

The current implementation is **model-agnostic for supported scikit-learn models**. A fresh clone does not contain a built-in production model; the user supplies a model bundle.

---

## 1. About the Project

A machine-learning model can work well during development and still behave differently after deployment. One common reason is **data drift**: the data reaching the model changes over time.

A basic monitoring system can tell us:

> “Several features are different from the reference data.”

But that still leaves a practical question:

> **“Which change is most likely the starting point?”**

DriftTrace addresses this by combining:

- reference-based statistical drift detection
- an optional dependency graph
- upstream root-cause tracing
- downstream symptom identification
- prediction explanations using SHAP and LIME
- a simple web UI
- human-controlled operational actions

The important distinction is that **drift detection and root-cause diagnosis are different steps**. DriftTrace first finds what changed, then uses dependency information to trace related changes upstream.

---

## 2. How DriftTrace Works

### Step 1 — Upload a model bundle

The current user contract is intentionally small:

```text
my_model.drift.zip
├── model.pkl
├── reference.csv
└── graph.json        # optional
```

### `model.pkl`
The trained, supported scikit-learn model.

### `reference.csv`
The reference or baseline data. DriftTrace uses this to understand what the model normally sees.

### `graph.json` (optional)
A declared dependency graph describing how important features are related.

A graph is **not required** for model loading, prediction, or basic drift detection. Without it, dependency-based upstream tracing is unavailable.

### Step 2 — Inspect and activate the model

DriftTrace inspects the uploaded model and identifies information such as:

- model framework
- task type
- feature names
- classification or regression behavior
- supported probability/output capabilities

The selected bundle becomes the active model used by the API.

### Step 3 — Make predictions

Predictions are served through FastAPI.

Each prediction can produce a standardized monitoring event containing information such as:

```text
timestamp
features
prediction / output
probability when available
model information
```

### Step 4 — Compare current data with the reference

DriftTrace compares recent observations against the reference baseline using statistical methods such as **KS-test and PSI**.

Conceptually:

```text
Reference data
      ↓
Current observations
      ↓
KS + PSI
      ↓
No drift / Drift detected
```

### Step 5 — Trace drift through the graph

When `graph.json` exists, DriftTrace checks which affected nodes are upstream or downstream of each other.

Example:

```text
mean_radius
      ↓
mean_perimeter
      ↓
mean_area
      ↓
prediction
```

If all three feature nodes are affected by the controlled scenario, the upstream node can be reported as the **root-cause candidate**, while the later nodes are retained as **symptoms**.

The graph is used for **tracing**, not for automatically propagating drift.

### Step 6 — Explain a prediction

For a specific prediction, DriftTrace can generate local explanations using **SHAP** or **LIME**.

These answer a different question from RCA:

> **RCA:** Where does the drift likely start?

> **SHAP/LIME:** Which features influenced this particular prediction?

### Step 7 — Human decision

DriftTrace does not silently replace a production model.

The operator reviews the evidence and decides what action should be taken, such as investigating the data, approving rollback, or approving retraining.

---

## 3. Installation and Setup

## System Requirements

### Backend

- Windows, Linux, or macOS
- Python **3.11 or newer**
- Python **3.13.5** has been used successfully for the current project
- 8 GB RAM recommended
- Git

### Frontend

- Node.js and npm
- A current Node.js LTS release is recommended

### Browser

Any modern Chrome, Edge, Firefox, or Safari browser.

### Optional

- Kafka/Redpanda can be used through the available streaming adapter, but it is **not required for the normal local demo**.

---

## Fresh Setup

### 1. Clone the repository

```bash
git clone https://github.com/shlokn678/DriftTrace.git
cd DriftTrace
```

### 2. Create a Python virtual environment

#### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install DriftTrace

```bash
python -m pip install --upgrade pip
pip install -e ".[serving,explain,dev,streaming]"
```

The project supports the core dependencies through `pyproject.toml` and keeps optional integrations separate from the main monitoring core.

### 4. Generate the local test-model bundles

The current repository contains tools for generating the five validation models locally:

```bash
python scripts/generate_test_models.py
```

This creates test bundles under:

```text
artifacts/test_models/
```

These generated bundles are runtime test artifacts and are intentionally not treated as normal source-code files.

### 5. Validate the test models

```bash
python scripts/validate_test_models.py
```

### 6. Start the backend

From the project root:

```bash
python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 7. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL shown by Vite, normally:

```text
http://localhost:5173/
```

---

## Quick Validation

Run the backend test suite:

```bash
python -m pytest -q
```

Run the API smoke workflow:

```bash
python scripts/smoke_api.py
```

The latest verified project state reports:

- **102 tests passed**
- **0 failed**
- **0 skipped**
- frontend TypeScript check passed
- frontend production build passed
- **15/15** smoke checks passed
- **5/5** generated model bundles passed validation

---

## 4. Core Features

### Model-agnostic onboarding

Users can supply different supported scikit-learn models instead of using one fixed domain model.

### Classification and regression

The current adapter has been validated with both classification and regression workflows.

### scikit-learn Pipelines

A preprocessing + estimator Pipeline can be treated as one deployable model.

### Reference-based drift detection

Every active model has its own reference dataset.

### KS-test

Measures how different two numerical distributions are.

### PSI

Measures how much the current population has shifted relative to the reference population.

### Categorical drift

Categorical features can be compared using frequency-based drift calculations.

### Output / prediction monitoring

Prediction output can also be monitored when enough output observations are available.

### Dependency-aware RCA

An optional graph lets DriftTrace trace affected nodes upstream and separate a likely origin from downstream symptoms.

### Graceful no-graph behavior

A model does not stop working just because it has no `graph.json`. Prediction and drift detection can still be used; dependency-based upstream tracing simply cannot be established.

### SHAP explanations

Explains how individual feature contributions influenced one prediction.

### LIME explanations

Creates a local approximation around one prediction and reports which features influenced it.

### Human-readable explanation layer

The UI converts raw SHAP/LIME contributions into short sentences such as:

> “The prediction was influenced most by Support calls, Engagement score, and Service score...”

The raw technical values remain available.

### Human-in-the-loop operation

Operational changes such as rollback or retraining are not performed silently by the system.

### Web UI

The React frontend displays:

- active model
- system status
- predictions
- drift status
- KS/PSI details
- root-cause information
- affected features
- SHAP/LIME explanations
- model switching
- technical diagnostic details

---

## 5. Current Tech Stack

| Layer | Technology | Current status |
|---|---|---|
| Language | Python | Current backend language |
| API | FastAPI | Implemented and verified |
| ML | scikit-learn | Current and tested adapter |
| Statistics | SciPy | KS-test and statistical calculations |
| Data | NumPy + pandas | Reference and monitoring data |
| Graphs | NetworkX | Dependency graph + RCA |
| Validation | Pydantic / PyYAML | API/config validation |
| Explainability | SHAP + LIME | Local prediction explanations |
| Frontend | React + TypeScript + Vite | Current UI |
| Version control | Git | Project source control |
| CI | GitHub Actions | Current validation workflow |
| Streaming adapter | Kafka API / Redpanda via `confluent-kafka` | Code present; not verified against a live broker in the current demo |

### Technologies not in the current active architecture

Some technologies appeared in the original project plan but were removed or deferred from the current final runtime:

- **MLflow** — removed/deferred from the current model-upload architecture
- **Apache Airflow** — removed/deferred from the current runtime
- **Docker** — removed from the current local-first architecture
- **Prometheus / Grafana** — future/stretch
- **AWS SageMaker** — future/stretch
- **Automatic graph learning / GNNs** — future research direction

This README intentionally separates completed functionality from planned extensions.

---

## 6. Simple Definitions of the Main Concepts

### Machine-learning model

A trained program that uses input data to produce a prediction or output.

**Here:** DriftTrace loads a supported trained scikit-learn model and serves its predictions.

### Model-agnostic

The monitoring system does not depend on one fixed set of feature names or one fixed business domain.

**Here:** different supported scikit-learn models can be uploaded using their own feature schemas and reference data.

### Reference data / baseline

A dataset representing what “normal” input data looked like for the model.

**Here:** `reference.csv` is used as the baseline for later drift comparisons.

### Data drift

A change in the distribution of data reaching a model.

**Example:** a feature that normally has one range of values starts receiving very different values.

**Here:** DriftTrace compares recent observations with the reference distribution.

### KS-test

The **Kolmogorov–Smirnov test** compares two numerical distributions.

It produces:

- a **KS statistic** — how different the distributions are
- a **p-value** — how strong the statistical evidence is that the distributions differ

A larger KS statistic means a larger distributional difference.

**Here:** DriftTrace uses KS as one part of its numerical drift decision.

### PSI

**Population Stability Index** measures how much a population has shifted compared with a reference population.

In simple terms:

> **How different is the current population from the baseline?**

**Here:** PSI is used alongside KS for numerical drift detection.

The current configuration treats PSI at or above **0.20** as drift evidence.

### P-value

A statistical measure used to judge how surprising the observed difference would be if there were no distribution change.

**Here:** the KS p-value is displayed so the operator can see the statistical evidence behind the drift result.

### Drifted feature

A feature whose recent distribution is sufficiently different from the reference distribution according to DriftTrace's drift rules.

### Dependency graph

A directed graph showing which features or nodes depend on other nodes.

Example:

```text
A → B → C → prediction
```

**Here:** `graph.json` optionally provides this information to the RCA engine.

### Root-cause candidate

The earliest or most upstream affected node in a declared dependency chain.

**Here:** if `A`, `B`, and `C` are drifting and the graph is `A → B → C`, `A` can be identified as the root-cause candidate for that controlled scenario.

Important: this is a **diagnostic candidate**, not proof that A physically caused B or C.

### Symptom

A downstream node that is affected in the same incident but is not the earliest affected node in the dependency path.

### Root-cause analysis (RCA)

The process of trying to identify where an observed problem most likely began.

**Here:** DriftTrace combines drift results with the declared graph to trace upstream.

### Prediction event

A record created around a model prediction so monitoring information can be collected.

**Here:** the event can contain the timestamp, input features, prediction/output, and model information.

### Monitoring window

A group of recent observations analyzed together instead of checking one prediction at a time.

**Here:** both paths exist — the batch/on-demand drift test and the continuous live monitor (Section 7a), which groups incoming events into tumbling windows (default 100 events) and evaluates each window with the same drift + RCA engines.

### SHAP

**SHapley Additive exPlanations** assigns contribution values to input features for a particular prediction.

Simple interpretation:

- positive contribution → pushes the explanation toward the selected output
- negative contribution → pushes it away
- larger magnitude → stronger influence in that explanation

**Here:** DriftTrace uses SHAP for **local prediction explanation**, not causal RCA.

### LIME

**Local Interpretable Model-agnostic Explanations** creates a simple local approximation around one prediction and uses it to estimate which features are influencing that prediction.

**Here:** LIME is another local prediction-explanation method available from the UI.

### Local explanation

An explanation for **one particular prediction**, not a summary of the entire model.

### Model adapter

A small interface that lets the generic DriftTrace core communicate with a particular ML framework.

**Here:** the implemented adapter is for scikit-learn.

### Classification

A task where the model predicts a class or category.

**Example:** Class 0 vs Class 1.

### Regression

A task where the model predicts a numerical value.

**Example:** predicting `55.76`.

### Pipeline

A packaged sequence of preprocessing + model steps treated as one deployable estimator.

**Here:** supported scikit-learn Pipelines can be uploaded as the model itself.

### Graceful degradation

The system continues to provide whatever functionality is available instead of inventing missing information.

**Here:** a model without `graph.json` can still predict and perform drift detection, but dependency-based upstream tracing is not available.

### Human-in-the-loop

The system provides evidence and recommendations while a person remains responsible for the operational decision.

**Here:** DriftTrace does not silently replace the active production model.

---

## 7. Current Validation Results

The latest implementation audit reports:

### Software quality

- **102 pytest tests passed**
- **0 test failures**
- **0 skipped tests**
- MyPy passed on **51 source files**
- Frontend TypeScript check passed
- Frontend production build passed
- API smoke workflow passed **15/15 checks**

### Model validation

Five generated validation bundles were tested:

1. Breast Cancer Logistic Regression — classification + graph
2. Wine Random Forest — classification, no graph
3. Digits Extra Trees — classification, no graph
4. Synthetic Gradient Boosting Classifier — classification + graph
5. Synthetic Gradient Boosting Regressor — regression, no graph

The latest audit reports classification test accuracy values in the **0.932–1.000** range and **R² = 0.956** for the regression validation model. These are validation-model results for testing the platform, not claims of production model performance.

### Drift and RCA validation

The strongest controlled scenarios produced:

- PSI values of approximately **8.7–12.4** against a **0.20** drift threshold
- extremely small KS p-values in the controlled drift scenarios
- stable scenarios with no meaningful drift
- graph-targeted scenarios with one upstream root-cause candidate and downstream symptoms

The clean single-root RCA result depends on injecting drift into the declared dependency chain. Drift does not automatically propagate through the graph; the graph is used for tracing related drift.

### Live monitoring

Continuous live monitoring is now implemented **in addition to** the batch/on-demand path
(see Section 7a). A background consumer groups incoming prediction events into tumbling
windows and runs the same KS + PSI drift engine and dependency-graph RCA per window, with a
consecutive-window incident lifecycle and persistent history:

```text
live predictions
      ↓
event stream (Redpanda topic, or broker-free local event log)
      ↓
tumbling window (default 100 events)
      ↓
EXISTING KS + PSI
      ↓
EXISTING RCA
      ↓
incident / alert / dashboard history
```

The verified end-to-end flow (NORMAL → STABLE windows → DRIFT → RCA root cause + symptoms →
OPEN incident → recovery) was demonstrated over the real HTTP API with the demo producer.
The broker-free file source is the default so the demo runs with no Kafka install; the
Redpanda broker path is available but a live broker is not exercised in the default demo.

---

## 7a. Live Monitoring (continuous)

DriftTrace now includes a **continuous live-monitoring path in addition to** the existing
batch/on-demand path. Both feed the **same** drift and RCA engines — there is one drift
implementation (KS + PSI) and one RCA implementation; the live layer only *orchestrates*
them for a continuously-arriving stream of prediction events.

```text
live prediction events
        ↓
Redpanda topic  (or broker-free local event log)
        ↓
monitoring consumer  (background thread)
        ↓
tumbling windows (default 100 events)
        ↓
EXISTING KS + PSI drift engine
        ↓
EXISTING dependency-graph RCA engine
        ↓
root candidate + downstream symptoms
        ↓
EXISTING root-cause alerting + incident lifecycle
        ↓
persistent monitoring history → live dashboard
```

### What a prediction event is

A standardized `PredictionEvent` (see `src/drifttrace/streaming/event.py`) carrying at
least: `event_id`, `timestamp`, `model_id` / `model_version`, input `features`, the
`prediction`/`output`, and optional `probability`. The same event the serving API already
emits after every `/predict` is reused — no new or conflicting schema was introduced.

### The deployed model stays active

Live monitoring runs against the **currently active model**. A new model is **not** created
per window. The reference baseline (`reference.csv`) remains the baseline for that model
version until the operator replaces the model or intentionally registers a new baseline.
DriftTrace never auto-updates the baseline.

### How tumbling windows work

Events are grouped into fixed-size, non-overlapping windows (default **100 events**,
configurable with `DRIFTTRACE_LIVE_WINDOW_SIZE`). When a window fills, DriftTrace runs the
existing KS + PSI engine against the active model's reference baseline, runs graph RCA when
a `graph.json` was provided, updates alerting/incident state, persists the window result,
and exposes it to the dashboard.

### Drift over time

Every evaluated window is persisted to `reports/live/windows.jsonl` with its `window_id`,
start/end timestamps, event count, drift status, per-feature KS statistic / KS p-value /
PSI, drifted features, RCA root candidate(s) and symptoms, and incident status. The
dashboard renders the progression (e.g. STABLE → STABLE → WARNING → DRIFT → DRIFT) and a
PSI / drifted-feature trend.

### Incident lifecycle (focused alerting, not spam)

A simple, configurable consecutive-window policy groups related drifted windows into **one
continuing incident** instead of a separate alert per window:

- `DRIFTTRACE_LIVE_INCIDENT_OPEN_AFTER` (default **2**) consecutive drifted windows → OPEN
- `DRIFTTRACE_LIVE_INCIDENT_CLOSE_AFTER` (default **3**) consecutive stable windows → CLOSE

This does **not** redefine drift. A window's drift verdict comes verbatim from the existing
drift engine; the policy only decides when a run of verdicts is treated as one incident.

### Graph RCA per window

For a graph-enabled model, each drifted window is traced with the existing RCA engine. For
the showcase chain `engagement_score → service_score → risk_indicator → prediction`, a
window where those nodes drift yields root candidate `engagement_score` with
`service_score` and `risk_indicator` as symptoms. No-graph models still detect drift but
report no symptom chain (`dependencies_available: false`); no causal relationship is
fabricated.

### Broker: optional and fault-tolerant

The default live source is **broker-free** (`DRIFTTRACE_LIVE_SOURCE=file`): the monitor
tails the local event log that `/predict` already writes, and `POST /live/ingest` lets a
producer push events over HTTP with no Kafka install. Set `DRIFTTRACE_LIVE_SOURCE=redpanda`
to consume from a Redpanda/Kafka topic. If the broker is unavailable, `/live/start` returns
a status of `ERROR` (surfaced in the UI) and the existing prediction and batch/on-demand
monitoring paths keep working.

### Live-monitoring API

| Endpoint | Purpose |
|---|---|
| `POST /live/start` | Start live monitoring for the active model (409 if none active) |
| `POST /live/stop` | Stop live monitoring |
| `GET /live/status` | Status (RUNNING/STOPPED/ERROR), counters, current window progress, latest result |
| `GET /live/windows?limit=N` | Persisted window history (drift over time) |
| `GET /live/latest` | Latest completed window result |
| `GET /live/incidents` | Open incident + closed incident history |
| `POST /live/ingest` | Broker-free event submission (demo/tests) |

The existing endpoints (`/predict`, `/demo/run-drift-test`, `/rca/latest`, `/models/*`,
`/explain`, …) are unchanged.

### Running the live demo

1. **(Optional) Start Redpanda** — only needed for `DRIFTTRACE_LIVE_SOURCE=redpanda`. The
   broker-free default needs nothing here. A common local option is the Redpanda container
   via `rpk`/`docker`; the normal DriftTrace demo does **not** require it:

   ```bash
   # optional, only for the broker-backed path
   rpk container start          # or your preferred local Redpanda
   # then run the backend with:  DRIFTTRACE_LIVE_SOURCE=redpanda
   ```

2. **Start the backend** (window size 100 is the default; smaller windows complete faster):

   ```bash
   python -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
   ```

3. **Start the frontend** and open the Live Monitoring section:

   ```bash
   cd frontend && npm run dev
   ```

4. **Activate a model** (the 8-feature graph bundle is the clearest showcase):

   Upload `artifacts/test_models/synthetic_gradient_boosting/synthetic_gradient_boosting.drift.zip`
   in the dashboard (or via `POST /models/upload`) and activate it.

5. **Start live monitoring** — click **Start** in the Live Monitoring section (or
   `POST /live/start`).

6. **Run the demo producer** — NORMAL traffic first, then switch to DRIFT:

   ```bash
   # stable traffic, then a drift burst on the declared chain, in one run:
   python scripts/live_demo_producer.py --mode normal --events 300 --then-drift 300 --interval 0.02
   ```

   Configurable: `--mode {normal,drift}`, `--events`, `--interval`, `--then-drift`,
   `--k` (shift in std units), `--seed`, `--transport {http,redpanda}`.

7. **Watch the dashboard**: stable windows appear first, then DRIFT windows with
   `engagement_score` as the root cause and `service_score`/`risk_indicator` as symptoms,
   an incident opens, and the PSI trend climbs. Returning to NORMAL traffic lets the
   incident close after the configured number of stable windows.

> **Honest demo note.** Clean *stable* windows require the producer's NORMAL traffic to
> reproduce the reference distribution faithfully and a window large enough for stable
> binning. The producer bootstraps whole reference rows for this reason. Very
> high-dimensional models with small windows can show incidental per-window drift from
> small-sample PSI binning variance — the 8-feature graph bundle with the default window
> of 100 gives the clearest NORMAL → DRIFT → RCA → incident progression.

---

## 8. Project Structure

```text
DriftTrace/
├── src/drifttrace/          # backend and core monitoring logic
├── frontend/                # React + TypeScript UI
├── tests/                   # automated tests
├── scripts/                 # setup, smoke and test-model utilities
├── config/                  # runtime configuration
├── docs/                    # project documentation
├── artifacts/               # generated runtime/test artifacts
├── reports/                 # generated reports
├── pyproject.toml           # Python package + dependencies
└── README.md                # this file
```

---

## 9. Important Limitations

1. **scikit-learn is the only implemented model adapter.** Other frameworks such as XGBoost, LightGBM, ONNX and PyTorch are future work.
2. **Live monitoring is implemented, but the default demo uses a broker-free source.** Continuous tumbling-window live monitoring (Section 7a) runs over a broker-free local event log by default; the Redpanda/Kafka adapter is wired in but a live broker is not exercised in the default demo. The batch/on-demand path remains fully available.
3. **RCA depends on graph quality.** A declared graph describes the dependency information available to the system; it does not automatically prove causal relationships.
4. **A clean one-root showcase requires controlled drift aligned with the declared graph chain.** Broad all-feature drift can naturally create multiple independent root candidates.
5. **The current project is local-first.** Production cloud deployment and dashboard infrastructure are future/stretch work.

---

## 10. Future Scope

- live broker-backed monitoring verified against a running Redpanda broker (the adapter and
  continuous tumbling-window pipeline are implemented; see Section 7a)
- sliding windows (in addition to the implemented tumbling windows)
- alert/dashboard integration with external systems
- broader model adapters
- Prometheus / Grafana integration
- cloud deployment such as AWS SageMaker
- automatic graph learning
- temporal dependency reasoning
- graph neural network approaches for future research

---

## Project Goal in One Line

> **DriftTrace turns “the model is drifting” into a more actionable question: “which related part changed first, what evidence supports that diagnosis, and what should the human operator investigate next?”**
