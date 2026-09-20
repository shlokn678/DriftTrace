# DriftTrace — Design & Architecture

**Scope of this document:** the complete system architecture, component responsibilities, data
flow, ML lifecycle, monitoring / drift / RCA flows, and deployment architecture. Tags from
`requirements.md` apply (`[PITCH]`, `[DERIVED]`, `[DECISION]`, `[STRETCH]`, `[FUTURE]`).

## 1. Architectural principles

1. **Pitch terminology is preserved.** Nodes are `income`, `credit_score`, `risk_score`,
   `prediction API`; the loop is detect -> trace -> flag root cause -> operator decides.
2. **Declared graph, not learned graph** `[PITCH]`. Causal discovery is explicitly out of scope.
3. **Core logic is broker-free and framework-free** `[DECISION]/[DERIVED]`. Drift, graph, and
   RCA are pure Python libraries. Airflow, Kafka, FastAPI, and Docker are thin adapters over
   them, so everything is unit-testable and laptop-friendly.
4. **Asynchronous monitoring** `[PITCH]`. Serving never blocks on drift computation.
5. **Root-cause-only alerting** `[PITCH]`. Symptoms are recorded, not alerted.
6. **Evidence everywhere** `[PITCH]`. Every run yields the five lifecycle-evidence artifacts.
7. **No invented externals** `[DERIVED]`. No credentials, cloud accounts, datasets, or paid APIs.

## 2. High-level system architecture

```
                        +--------------------------------------------------+
                        |                  DriftTrace (MVP)                |
                        +--------------------------------------------------+

  DATA & VERSIONING            BUILD & TRACK                 ORCHESTRATE
  +------------------+         +-------------------+         +----------------------+
  | data generator   |         | feature pipeline  |         | Airflow DAG          |
  | schema validator |  --->   | model training    |  --->   | ingest -> validate   |
  | DVC (local remote)|        | evaluation        |         |  -> drift-check       |
  +------------------+         | MLflow tracking + |         |  -> report / retrain |
          |                    | model registry    |         | (CLI-callable tasks) |
          v                    +-------------------+         +----------------------+
  Git (code/schema/pointers)            |                              |
                                        v                              |
  SERVE (FastAPI + Docker)      declared dependency graph              |
  +---------------------+        (graph.yaml -> NetworkX)              |
  | /health /ready      |                |                            |
  | /predict /model-info|                v                            v
  | /explain /rca/latest|        MONITOR (async)                RESPONSIBLE AI
  | request logging     | <----  +---------------------+       +------------------+
  +----------+----------+        | event source:       |       | SHAP / LIME      |
             |                   |  Kafka/Redpanda OR   |       | fairness checks  |
             | prediction events |  file-replay        |       | privacy checks   |
             +-----------------> | windowing           |       | governance list  |
                                 | drift (KS/PSI)      |       +------------------+
                                 | RCA (graph upstream)|
                                 | root-cause alert -> webhook stub + logs + report
                                 +---------------------+

  CI/CD: GitHub Actions (lint, type, unit, schema, image build, compose smoke test)

  STRETCH (not MVP-mandatory):
    Prometheus + Grafana (scrape /metrics, dashboards, Grafana alerting)  [STRETCH]
    AWS SageMaker deployment/monitoring                                    [STRETCH]
```

## 3. Component responsibilities

Core library package: `src/drifttrace/`. Every subpackage is importable and testable on its own.

| Component | Package | Responsibility | Requirements |
| --- | --- | --- | --- |
| Data generator | `data/generator.py` | Deterministic seeded synthetic loan-default dataset with chained features. | FR-1.4, NFR-12 |
| Schema & validator | `data/schema.py`, `data/validate.py` | Declared column schema; type/null/range checks; validation report. | FR-2 |
| DVC integration | `data/versioning.py` + `dvc.yaml` | Track data/model artifacts; local remote; pointer files in Git. | FR-1 |
| Feature pipeline | `features/transform.py` | Compute chained features; shared by train and serve; no skew. | FR-3 |
| Graph definition | `config/graph.yaml`, `graph/loader.py`, `graph/dag.py` | Declared DAG loaded into NetworkX; parents/children/ancestors; validation. | FR-8, FR-3.4 |
| Training & eval | `training/train.py`, `training/evaluate.py` | Deterministic split, fit, metrics, gate, persist artifact. | FR-4 |
| Tracking & registry | `training/registry.py` | MLflow runs, params/metrics/artifacts, model versions, audit trail. | FR-5, FR-15.3 |
| Drift detection | `drift/ks.py`, `drift/psi.py`, `drift/detector.py` | Per-node KS/PSI vs training baseline; verdicts. | FR-9 |
| Baseline store | `drift/baseline.py` | Versioned training reference distributions per model version. | FR-9.2/9.3 |
| RCA engine | `rca/engine.py` | Trace upstream; earliest drifted node(s); symptom path; evidence. | FR-10 |
| Alerting | `alerting/alerter.py`, `alerting/webhook.py` | Root-cause-only alert, cool-down, webhook + local persistence. | FR-11 |
| Reports | `governance/report.py` | JSON + human-readable monitoring report with lifecycle evidence. | FR-17 |
| Serving API | `serving/app.py`, `serving/schemas.py` | FastAPI endpoints, model load by version, request logging, /metrics. | FR-7, FR-19.3 |
| Streaming monitor | `streaming/source.py`, `streaming/window.py`, `streaming/monitor.py` | Event source interface (Kafka/Redpanda + file-replay), windowing, async loop. | FR-13 |
| Explainability | `explain/shap_explainer.py`, `explain/lime_explainer.py` | Local + global explanations by model version. | FR-14 |
| Responsible AI | `governance/fairness.py`, `governance/privacy.py`, `docs/governance-checklist.md` | Fairness metrics, PII deny-list checks, governance checklist. | FR-15 |
| Drift-injection harness | `data/inject.py` | Seeded scenarios incl. monthly->annual income. | FR-18 |
| Operator CLI | `cli/main.py` | `run-pipeline`, `serve`, `monitor`, `rca show`, `rollback`, `retrain --approve`, `inject`. | FR-6.4, FR-12 |
| Orchestration | `pipelines/airflow/dags/drifttrace_dag.py` | Thin DAG wrapping CLI/library tasks; approval gate. | FR-6 |

## 4. The declared dependency graph `[PITCH]/[DECISION]`

`config/graph.yaml` is the single source of truth for both feature computation order and RCA
traversal (FR-3.4, FR-8.4). Illustrative shape (final content decided at implementation time):

```yaml
# graph.yaml  — declared feature dependency DAG
nodes:
  income:        { kind: raw_input,      drift: { psi: true, ks: true } }
  credit_score:  { kind: derived_feature, parents: [income],       drift: { psi: true, ks: true } }
  risk_score:    { kind: derived_feature, parents: [credit_score], drift: { psi: true, ks: true } }
  prediction:    { kind: model_output,    parents: [risk_score] }
edges:
  - [income, credit_score]
  - [credit_score, risk_score]
  - [risk_score, prediction]
```

The loader builds a NetworkX `DiGraph`, rejects cycles/undefined parents/orphans (FR-8 AC-2),
and exposes `parents(n)`, `children(n)`, `ancestors(n)`.

## 5. Data flow

### 5.1 Batch / training data flow (ML lifecycle stages 1-3)
```
generator (seed) -> raw dataset -> DVC track + Git pointer
      -> schema validate -> feature transform (chained) -> train/val/test split
      -> fit model -> evaluate (metrics + fairness) -> gate
      -> MLflow log (params, metrics, artifact, dataset version, commit)
      -> if pass: register model version + persist drift baseline for that version
```

### 5.2 Serving data flow (lifecycle stage 5)
```
client -> POST /predict -> load registered model version -> feature transform (same code)
      -> predict -> log request (features, prediction, model version, timestamp)
      -> emit prediction event to stream (fire-and-forget)  [async, non-blocking]
      -> return prediction + model version
```

### 5.3 Monitoring data flow (lifecycle stage 6)
```
event source (Redpanda consumer OR file-replay) -> window assembler
      -> on window close:
           per-node drift (KS/PSI vs baseline of the deployed model version)
           -> RCA (trace upstream on declared graph)
           -> monitoring report (JSON + summary)
           -> if root cause AND not in cool-down: alert (webhook stub + logs)
```

## 6. ML lifecycle mapping `[PITCH]`

| Stage | Pitch label | DriftTrace realization |
| --- | --- | --- |
| 1 | Data: collect, validate, version | generator + schema validator + DVC |
| 2 | Build: engineer features, train | feature transform + training/evaluation |
| 3 | Track: log runs, register model | MLflow tracking + model registry |
| 4 | Automate: DAG + CI/CD tests | Airflow DAG + GitHub Actions |
| 5 | Deploy: API + container | FastAPI + Docker (+ Compose) |
| 6 | Monitor: drift, RCA, alerts | streaming monitor + drift + RCA + alerting |

Lifecycle evidence per run (FR-17): dataset version (DVC), code commit (Git), pipeline execution
(Airflow run id / CLI run id), model artifact (MLflow), monitoring report (reports/).

## 7. Drift detection flow (FR-9)

```
for each node in graph:
   window_values = values for node in current window
   if len(window_values) < min_samples:  verdict = INSUFFICIENT_DATA
   else:
      continuous -> KS-test(window vs baseline)  (SciPy)  -> p-value
      continuous(binned)/categorical -> PSI(window vs baseline)
      verdict = DRIFT / WARNING / STABLE per configured thresholds
   record: statistic, p-value, PSI, thresholds, verdict, window id, baseline id
```
KS p < 0.05 -> drift; PSI >= 0.2 -> drift, 0.1-0.2 -> warning, < 0.1 -> stable (per-node
configurable). The baseline is the training reference distribution of the deployed model version.

## 8. Root-cause-analysis flow (FR-10)

```
drifted = { nodes with verdict DRIFT }
if drifted is empty: report "no root cause"; emit no alert

for each node in drifted:
   if node has NO drifted ancestor in the declared graph:
       mark node as ROOT-CAUSE CANDIDATE
   else:
       mark node as SYMPTOM
symptom_path(candidate) = drifted descendants reachable from candidate toward the output
if multiple candidates: rank by drift severity, report all (co-equal)   [DECISION FR-10.5]
attach per-node evidence to every node on each symptom path
```
Worked example (matches the pitch): drift at `income`, `credit_score`, `risk_score` ->
`income` is the single candidate; `credit_score` and `risk_score` are symptoms on the path to
`prediction API`.

## 9. Monitoring flow and asynchrony (FR-13, NFR-2)

- Serving emits prediction events fire-and-forget; the request path never waits for the monitor.
- The monitor is a separate process/container consuming the stream, assembling tumbling windows
  (size + min-samples configurable), and running drift + RCA per closed window.
- Event source is an interface: `RedpandaSource` (Kafka API) and `FileReplaySource`. CI and unit
  tests use file-replay; identical results are asserted for the same event sequence (FR-13.6/AC-5).
- Consumer offsets are committed so restart resumes without reprocessing (FR-13 AC-3).

## 10. Alerting flow (FR-11)

```
rca_result -> if has root cause:
   key = (root_cause_node, model_version)
   if key active within cool-down: suppress
   else:
      payload = { root_cause, symptom_path, evidence, model_version, window_id }
      persist alert locally (always)
      POST payload to webhook stub; on failure log + keep local record (no crash)
```
Exactly one alert per causal chain per window; symptoms recorded in the report only.

## 11. Deployment architecture (MVP) `[PITCH]`

Docker Compose services (broker-free and full profiles):

| Service | Image basis | Purpose | Profile |
| --- | --- | --- | --- |
| `api` | project image | FastAPI serving + /metrics | all |
| `mlflow` | mlflow | tracking + registry UI (local backend) | all |
| `webhook-stub` | project image | receives root-cause alerts locally | all |
| `monitor` | project image | async streaming monitor | full |
| `redpanda` | redpanda | Kafka-API broker | full |
| `airflow` | airflow | DAG orchestration (optional locally) | orchestrate |
| `prometheus` | prometheus | scrape /metrics | stretch |
| `grafana` | grafana | dashboards + alerting | stretch |

- **Profiles** `[DECISION]`: `core` (api + mlflow + webhook-stub, < ~2 GB), `full` (adds
  redpanda + monitor), `stretch` (adds prometheus + grafana). Supports NFR-3.
- Single project image built from `docker/Dockerfile`; entrypoint selects api / monitor / cli.
- No host-specific paths; config via env vars with safe local defaults (NFR-4, NFR-7).

## 12. Deployment architecture (stretch) `[STRETCH]`

- Prometheus scrapes `api` and `monitor` `/metrics`; Grafana dashboards for drift/RCA/health;
  Grafana alerting mirrors the webhook root-cause alert (FR-19).
- AWS SageMaker: the same container image + MLflow model artifact deployed to a SageMaker
  endpoint with SageMaker Model Monitor, no change to core drift/RCA logic (FR-20). No AWS
  account/role/credential is configured now.

## 13. Configuration model `[DECISION]`

- `config/graph.yaml` — declared dependency graph.
- `config/schema.yaml` — data schema for validation.
- `config/drift.yaml` — per-node thresholds, window policy, min-samples, cool-down.
- `config/governance.yaml` — sensitive attribute, PII deny-list, retention note.
- `.env.example` — non-secret env defaults (ports, MLflow URI, broker address). No real secrets.

## 14. Error handling & resilience

- Graph load failure, schema failure, and below-gate models fail fast with named causes.
- Webhook failure never crashes the monitor (FR-11 AC-4).
- Insufficient window data yields a non-alerting verdict (FR-9 AC-4).
- Monitor slowness never affects prediction latency (NFR-2).

## 15. Security & privacy posture `[DERIVED]/[DECISION]`

- MVP API unauthenticated, localhost/internal-network only; recorded limitation, not public-safe
  (FR-7.6).
- No PII in logs/artifacts/events; automated privacy check enforces the deny-list (FR-15.5, NFR-9).
- No invented credentials or external services (NFR-7).

## 16. Traceability
Requirements: `requirements.md`. Phased tasks: `tasks.md`. Syllabus units: `syllabus-mapping.md`.
Decisions: `assumptions-and-decisions.md`.
