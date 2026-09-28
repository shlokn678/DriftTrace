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

**Here:** the current verified demo primarily uses controlled/on-demand monitoring scenarios; continuous broker-backed monitoring is a planned next extension.

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

### Live monitoring limitation

The project contains streaming/Redpanda adapter code, but the current verified demo is **on-demand/batch rather than always-on live broker monitoring**.

The next planned extension is:

```text
live predictions
      ↓
event stream
      ↓
sliding / tumbling window
      ↓
automatic KS + PSI
      ↓
RCA
      ↓
alert / dashboard
```

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
2. **Continuous broker-backed monitoring is not the current demo path.** The current verified monitoring flow is on-demand/batch.
3. **RCA depends on graph quality.** A declared graph describes the dependency information available to the system; it does not automatically prove causal relationships.
4. **A clean one-root showcase requires controlled drift aligned with the declared graph chain.** Broad all-feature drift can naturally create multiple independent root candidates.
5. **The current project is local-first.** Production cloud deployment and dashboard infrastructure are future/stretch work.

---

## 10. Future Scope

- continuous Kafka/Redpanda monitoring with sliding/tumbling windows
- always-on asynchronous drift monitoring
- alert/dashboard integration
- broader model adapters
- Prometheus / Grafana integration
- cloud deployment such as AWS SageMaker
- automatic graph learning
- temporal dependency reasoning
- graph neural network approaches for future research

---

## Project Goal in One Line

> **DriftTrace turns “the model is drifting” into a more actionable question: “which related part changed first, what evidence supports that diagnosis, and what should the human operator investigate next?”**
