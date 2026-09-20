# DriftTrace — Requirements

**Project:** DriftTrace — An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift
**Course:** CI3203D · Machine Learning and Operations
**Programme:** Computer Science & Engineering (Artificial Intelligence)
**Spec status:** Planning stage. No application code implemented yet.

## 0. How to read this document

The project pitch is the single source of truth. Every statement below is tagged so that
pitch content is never confused with engineering interpretation:

| Tag | Meaning |
| --- | --- |
| `[PITCH]` | Stated explicitly in the pitch deck. Wording and terminology preserved. |
| `[DERIVED]` | A direct, mechanical consequence of a `[PITCH]` statement (no new scope). |
| `[DECISION]` | The pitch leaves this unspecified. A reasonable choice is proposed and recorded in `assumptions-and-decisions.md`. Open to change. |
| `[STRETCH]` | The pitch explicitly calls this a stretch goal. Not required for the MVP. |
| `[FUTURE]` | The pitch lists this under Future Scope. Out of scope for this course project. |

Untagged text is structural.

---

## 1. Problem statement `[PITCH]`

> A model can be accurate in training and still fail in production.

Three failure modes motivate the project:

1. **Data and feature changes.** An upstream transformation changes `income` from monthly to
   annual, silently shifting every downstream feature.
2. **Independent drift alarms.** Existing monitors test features separately and may report
   12 drifted features instead of identifying the original failure.
3. **No operational response.** Without versioning, pipelines, APIs, logging, and alerts,
   engineers cannot reproduce, diagnose, or safely maintain the model.

**MLOps objective `[PITCH]`:** make the complete ML lifecycle reproducible, automated,
deployable, observable, and maintainable.

## 2. Proposed solution `[PITCH]`

> DriftTrace turns a wall of alerts into one actionable diagnosis.

The four-step solution loop, exactly as pitched:

1. Detect drift per node using KS-test / PSI.
2. Read the declared dependency graph and trace upstream.
3. Flag the earliest drifted node; log and alert only on the root-cause candidate.
4. Let operators inspect the evidence and decide whether to fix data, roll back, or retrain.

## 3. Application / use case `[PITCH]`

A production **loan-default / churn** pipeline with **chained features** and a **deployed
prediction API**. Finance use case (Unit VI).

The pitch worked example, with node roles preserved:

| Node | Role in the pitch |
| --- | --- |
| `income` | **ROOT CAUSE** |
| `credit_score` | **SYMPTOM** |
| `risk_score` | **SYMPTOM** |
| `prediction API` | **OUTPUT** |

The dependency chain is `income -> credit_score -> risk_score -> prediction API`.

## 4. MLOps lifecycle coverage `[PITCH]`

| # | Stage | Pitch description |
| --- | --- | --- |
| 1 | Data | Collect, validate, version |
| 2 | Build | Engineer features, train |
| 3 | Track | Log runs, register model |
| 4 | Automate | DAG + CI/CD tests |
| 5 | Deploy | API + container |
| 6 | Monitor | Drift, RCA, alerts |

**Lifecycle evidence `[PITCH]`:** each run is reproducible from a versioned dataset, a code
commit, a pipeline execution, a model artifact, and a monitoring report. These five artifacts
define a reproducible run throughout this spec.

## 5. Scope: MVP versus stretch `[PITCH]`

The pitch states: *"MVP deployment: FastAPI + Docker Compose + MLflow + GitHub Actions.
Stretch: Prometheus/Grafana and AWS SageMaker."*

| Capability | Stack | Scope |
| --- | --- | --- |
| Version & validate | Git + DVC | MVP |
| Track & register | MLflow | MVP |
| Orchestrate | Apache Airflow | MVP |
| Serve | FastAPI + Docker | MVP |
| CI/CD | GitHub Actions | MVP |
| Drift detection + RCA + alerts | KS-test / PSI + NetworkX + webhook | MVP |
| Streaming drift | Kafka / Redpanda | MVP (roadmap Phase 3) |
| Explainability | SHAP / LIME | MVP (Unit V) |
| Responsible AI checks | Fairness, privacy, governance checklist | MVP (Unit V) |
| Monitor dashboards & alerting | Prometheus + Grafana | **Stretch** |
| Cloud deployment & monitoring | AWS SageMaker | **Stretch** |
| Automatic graph learning, temporal windows + GNNs | - | **Future** |

Two scope notes that resolve apparent tension inside the pitch:

- Prometheus/Grafana appears in the deployment stack and in roadmap Phase 4 ("Operate:
  Prometheus/Grafana + RCA"), but the MVP line names it as stretch. **RCA is MVP-mandatory;
  Prometheus/Grafana is stretch.** `[DERIVED]` For MVP alerting the project uses the other
  channel the pitch already names: *"Alert via Grafana / webhook"* -> webhook. `[DERIVED]`
- Kafka is not in the MVP deployment sentence but is named in roadmap Phase 3
  ("Deploy: FastAPI + Docker + Kafka") and on the real-time monitoring slide. **Streaming
  stays in scope for Phase 3.** `[PITCH]` A file-replay event source is added so drift logic
  can be tested without a running broker. `[DECISION]`

---

## 6. Functional requirements

### FR-1 - Dataset collection and versioning
**User story:** As an ML engineer, I want every dataset version tracked, so that any training
run or drift report can be reproduced from an exact data snapshot.

- FR-1.1 `[PITCH]` The system SHALL version datasets with DVC.
- FR-1.2 `[PITCH]` Git SHALL track code, schema, and DVC pointer files; DVC SHALL track data
  and model artifacts.
- FR-1.3 `[DECISION]` The DVC remote for the MVP SHALL be a local directory in the workspace
  (no cloud bucket, no credentials).
- FR-1.4 `[DECISION]` The loan-default dataset SHALL be produced by a deterministic, seeded
  local generator that materialises the chained features
  (`income -> credit_score -> risk_score`). No external dataset is downloaded and no external
  API is called.

**Acceptance criteria**
1. WHEN a dataset is generated or updated THEN the system SHALL produce a DVC-tracked artifact
   whose content hash is recorded in a Git-committed pointer file.
2. WHEN a previous dataset version is requested by its Git commit THEN `dvc checkout` SHALL
   restore byte-identical data.
3. WHEN the generator runs twice with the same seed and parameters THEN the outputs SHALL be
   byte-identical.

### FR-2 - Data validation and schema checks
**User story:** As an ML engineer, I want incoming data validated against a declared schema,
so that malformed data is rejected before training or serving.

- FR-2.1 `[PITCH]` The system SHALL perform validation checks on data (Unit II) and schema
  tests in CI (Unit III / CI/CD).
- FR-2.2 `[DERIVED]` The schema SHALL declare, per column, data type, nullability, and an
  allowed range or category set.
- FR-2.3 `[DERIVED]` Validation results SHALL be recorded as part of the pipeline execution
  evidence.
- FR-2.4 `[DECISION]` Validation SHALL be a dependency-light schema module (a declared schema
  file plus a validator) rather than a heavyweight data-quality framework, to keep the laptop
  footprint small.

**Acceptance criteria**
1. WHEN a batch violates a declared type, nullability, or range rule THEN validation SHALL fail
   and SHALL name every offending column and rule.
2. WHEN a batch satisfies the schema THEN validation SHALL pass and emit a validation report.
3. IF validation fails inside the orchestrated pipeline THEN downstream training SHALL NOT run.
4. Note: a monthly->annual `income` change is schema-valid in general; it is a **drift** concern
   (FR-9), not a schema concern. Validation SHALL NOT be relied on to catch it.

### FR-3 - Feature engineering with declared chained features
**User story:** As an ML engineer, I want the feature chain expressed as code with declared
dependencies, so that drift in one feature can be traced to its upstream source.

- FR-3.1 `[PITCH]` The system SHALL engineer features including the chained features
  `income -> credit_score -> risk_score`.
- FR-3.2 `[PITCH]` Feature dependencies SHALL be **declared** ("read the declared dependency
  graph"), not inferred.
- FR-3.3 `[DERIVED]` The same transformation code SHALL be used for training and serving, so
  training/serving skew is not introduced by duplicate logic.
- FR-3.4 `[DECISION]` The declaration SHALL live in a single version-controlled graph
  definition file that is the one source of truth for feature computation order and RCA
  traversal.

**Acceptance criteria**
1. WHEN the feature pipeline runs THEN every feature named in the graph SHALL be produced, and
   no feature outside it SHALL be produced.
2. WHEN a feature is added in code without a matching graph declaration THEN a consistency test
   SHALL fail.
3. WHEN the same raw record is transformed by the training path and the serving path THEN the
   resulting feature vectors SHALL be equal.

### FR-4 - Model training and evaluation
**User story:** As an ML engineer, I want reproducible training and evaluation, so that model
quality is measured consistently across versions.

- FR-4.1 `[PITCH]` The system SHALL train and evaluate a model for the loan-default / churn
  task inside the automated pipeline.
- FR-4.2 `[DERIVED]` Train/validation/test splitting SHALL be deterministic and recorded.
- FR-4.3 `[DECISION]` The MVP model SHALL be a scikit-learn classifier (logistic-regression
  baseline plus one gradient-boosted-tree candidate). Deep learning is not required and is
  excluded from the MVP.
- FR-4.4 `[DECISION]` Reported metrics SHALL be ROC-AUC (primary), PR-AUC, accuracy, precision,
  recall, F1.
- FR-4.5 `[DERIVED]` Training SHALL persist the fitted pipeline as a single loadable artifact.

**Acceptance criteria**
1. WHEN training runs twice on the same dataset version and seed THEN reported metrics SHALL be
   identical.
2. WHEN training completes THEN a model artifact plus a metrics record SHALL be written and
   logged.
3. IF a candidate scores below the configured minimum ROC-AUC THEN the run SHALL be marked
   failed and the model SHALL NOT be registered.

### FR-5 - Experiment tracking and model registry
**User story:** As an ML engineer, I want runs and model versions catalogued, so that I can
compare experiments and roll back to a known-good model.

- FR-5.1 `[PITCH]` The system SHALL use MLflow to log parameters, metrics, and artifacts and to
  manage model versions.
- FR-5.2 `[DERIVED]` Every run SHALL record the dataset version, Git commit, and pipeline
  execution id, completing the lifecycle-evidence chain.
- FR-5.3 `[PITCH]` The MLflow/DVC record SHALL serve as the governance **audit trail**.
- FR-5.4 `[DECISION]` MLflow SHALL run locally with a file/SQLite backend and a local artifact
  root. No hosted server or account is assumed.

**Acceptance criteria**
1. WHEN a training run finishes THEN MLflow SHALL contain one run with parameters, metrics, the
   model artifact, the dataset version, and the code commit.
2. WHEN a model passes evaluation gates THEN it SHALL be registered as a new version.
3. WHEN a registered version is requested THEN serving SHALL load exactly that version.

### FR-6 - Pipeline orchestration
**User story:** As an ML engineer, I want the lifecycle steps orchestrated as a DAG, so that
the pipeline runs on a schedule and on demand without manual steps.

- FR-6.1 `[PITCH]` The system SHALL provide an Apache Airflow DAG implementing
  `ingest -> validate -> drift-check -> report / retrain`.
- FR-6.2 `[PITCH]` Retraining SHALL be **human-approved** (FR-12); the DAG SHALL NOT silently
  push a new model to production.
- FR-6.3 `[DERIVED]` Task failures SHALL stop downstream tasks and be visible in the Airflow UI
  with logs.
- FR-6.4 `[DECISION]` Every DAG task SHALL be a thin wrapper over a library function callable
  from the CLI, so the whole pipeline runs without Airflow (for CI and low-RAM laptops).

**Acceptance criteria**
1. WHEN the DAG is triggered THEN tasks SHALL execute in order
   `ingest -> validate -> drift-check -> report / retrain`.
2. IF validation fails THEN drift-check and report/retrain SHALL NOT run.
3. WHEN drift-check detects drift THEN the report branch SHALL always run and the retrain branch
   SHALL only be reachable through an explicit human approval gate.
4. WHEN the pipeline is invoked via CLI instead of Airflow THEN it SHALL produce the same
   artifacts.

### FR-7 - Prediction API (serving)
**User story:** As a consumer of the model, I want a REST prediction endpoint with health
checks, so that predictions are available to applications and operators.

- FR-7.1 `[PITCH]` The system SHALL expose FastAPI REST endpoints with health checks.
- FR-7.2 `[PITCH]` Predictions SHALL remain low-latency while monitoring runs asynchronously.
- FR-7.3 `[DERIVED]` The API SHALL load a specific registered model version and report it.
- FR-7.4 `[DERIVED]` Every prediction request SHALL be logged with input features, prediction,
  model version, and timestamp (logging is a named missing capability in the problem statement).
- FR-7.5 `[DECISION]` MVP endpoints: `GET /health`, `GET /ready`, `POST /predict`,
  `GET /model-info`, `POST /explain` (FR-14), `GET /rca/latest` (FR-10/11).
- FR-7.6 `[DECISION]` The MVP API is unauthenticated and intended for localhost / internal
  Compose network only. This is recorded as an explicit security limitation in
  `assumptions-and-decisions.md` and must not be exposed publicly as-is.

**Acceptance criteria**
1. WHEN `GET /health` is called THEN the service SHALL return HTTP 200 while live.
2. WHEN `GET /ready` is called before a model is loaded THEN the service SHALL return a
   non-ready status rather than a successful prediction.
3. WHEN a well-formed request hits `POST /predict` THEN the response SHALL contain the
   prediction and the serving model version.
4. WHEN a malformed request hits `POST /predict` THEN the service SHALL return HTTP 422 with a
   field-level error and SHALL NOT crash.
5. WHEN monitoring is under load THEN prediction latency SHALL NOT be blocked by it (NFR-2).

### FR-8 - Declared dependency graph
**User story:** As an operator, I want the feature pipeline represented as a graph, so that
drift can be traced from a symptom back to its source.

- FR-8.1 `[PITCH]` The system SHALL represent the declared dependency graph using NetworkX.
- FR-8.2 `[PITCH]` The graph SHALL contain the chain
  `income -> credit_score -> risk_score -> prediction API`.
- FR-8.3 `[DERIVED]` The graph SHALL be a DAG; each node SHALL carry its kind (raw input,
  derived feature, model output) and its drift-test configuration.
- FR-8.4 `[DECISION]` The graph SHALL be declared in a human-editable YAML file loaded into
  NetworkX at runtime, so operators can extend the pipeline without code changes.

**Acceptance criteria**
1. WHEN the graph file is loaded THEN the system SHALL build a NetworkX `DiGraph` whose nodes
   and edges match the file exactly.
2. IF the graph contains a cycle, an undefined parent, or an orphan node THEN loading SHALL fail
   naming the offending nodes.
3. WHEN a node is queried THEN the system SHALL return its direct parents, direct children, and
   all transitive upstream ancestors.

### FR-9 - Per-node drift detection
**User story:** As an operator, I want drift measured for every node against the training
baseline, so that I know which parts of the pipeline have changed.

- FR-9.1 `[PITCH]` The system SHALL detect drift **per node** using the KS-test and PSI.
- FR-9.2 `[PITCH]` The reference baseline SHALL come from training data.
- FR-9.3 `[DERIVED]` The baseline SHALL be versioned alongside the model version it belongs to.
- FR-9.4 `[DECISION]` KS-test applies to continuous nodes; PSI to continuous (binned) and
  categorical nodes. Default thresholds: KS p-value < 0.05 -> drift; PSI >= 0.2 -> drift,
  0.1 <= PSI < 0.2 -> warning, PSI < 0.1 -> stable. Configurable per node.
- FR-9.5 `[DECISION]` A minimum window sample size SHALL be enforced; smaller windows are
  reported as "insufficient data" rather than drift.

**Acceptance criteria**
1. WHEN a window is compared against the training baseline THEN the system SHALL output, per
   node, the KS statistic and p-value, the PSI value, the thresholds used, and a
   drift/warning/stable verdict.
2. WHEN a node distribution is unchanged THEN that node SHALL be reported stable.
3. WHEN `income` is multiplied by 12 in the incoming window THEN `income` SHALL be reported
   drifted.
4. WHEN a window has fewer than the minimum samples THEN the verdict SHALL be "insufficient
   data" and no alert SHALL be raised.

### FR-10 - Root-cause analysis
**User story:** As an operator, I want one diagnosis instead of a wall of alerts, so that I can
act on the original failure rather than its symptoms.

- FR-10.1 `[PITCH]` The system SHALL read the declared dependency graph and trace upstream.
- FR-10.2 `[PITCH]` The system SHALL flag the **earliest drifted node** as the root-cause
  candidate.
- FR-10.3 `[PITCH]` The system SHALL report the **downstream symptom path** from the root cause
  to the affected output.
- FR-10.4 `[DERIVED]` Nodes drifted but having a drifted ancestor SHALL be classified as
  symptoms, not root causes.
- FR-10.5 `[DECISION]` If several drifted nodes have no drifted ancestor, all SHALL be reported
  as co-equal root-cause candidates ranked by drift severity, rather than guessing one winner.
- FR-10.6 `[DERIVED]` RCA output SHALL carry per-node evidence (statistics, thresholds, window
  id, baseline id) so operators can *inspect the evidence*.

**Acceptance criteria**
1. WHEN `income`, `credit_score`, and `risk_score` are all drifted THEN the system SHALL report
   `income` as root-cause candidate and the other two as downstream symptoms.
2. WHEN only `risk_score` is drifted THEN `risk_score` SHALL be the root-cause candidate.
3. WHEN no node is drifted THEN the system SHALL report no root cause and emit no alert.
4. WHEN an RCA result is produced THEN it SHALL include drift evidence for every node on the
   reported symptom path.

### FR-11 - Root-cause-only logging and alerting
**User story:** As an operator, I want to be alerted once, about the root cause, so that alert
fatigue does not hide the real failure.

- FR-11.1 `[PITCH]` The system SHALL log and alert **only on the root-cause candidate**.
- FR-11.2 `[PITCH]` Alerting SHALL be delivered via Grafana / webhook. `[DERIVED]` MVP uses the
  webhook path plus structured local logs; Grafana alerting arrives with the stretch work.
- FR-11.3 `[DERIVED]` Symptom-node drift SHALL still be recorded in the report as supporting
  evidence but SHALL NOT generate its own alert.
- FR-11.4 `[DECISION]` Repeat alerts for an unchanged, still-open root cause SHALL be suppressed
  for a configurable cool-down period.
- FR-11.5 `[DECISION]` The MVP webhook receiver SHALL be a local stub service in the Compose
  stack. No third-party notification service, token, or account is assumed.

**Acceptance criteria**
1. WHEN 3 nodes are drifted on one causal chain THEN exactly 1 alert SHALL be emitted, naming
   the root-cause candidate.
2. WHEN an alert is emitted THEN its payload SHALL contain the root-cause node, symptom path,
   drift evidence, model version, and window id.
3. WHEN the same root cause persists across consecutive windows inside the cool-down period THEN
   no duplicate alert SHALL be emitted.
4. WHEN the webhook endpoint is unreachable THEN the alert SHALL still be persisted locally and
   the failure logged without crashing the monitor.

### FR-12 - Operator response and human-approved action
**User story:** As an operator, I want to inspect the evidence and choose the response, so that
no automated system retrains or rolls back on my behalf.

- FR-12.1 `[PITCH]` Operators SHALL inspect the evidence and decide whether to fix data, roll
  back, or retrain.
- FR-12.2 `[PITCH]` Rollback and retraining SHALL be **human-approved**.
- FR-12.3 `[DERIVED]` Rollback SHALL mean pointing serving at a previously registered model
  version from the MLflow registry.
- FR-12.4 `[DECISION]` The MVP surface for inspection and approval SHALL be the RCA report plus
  CLI commands (`drifttrace rca show`, `drifttrace rollback --to-version`,
  `drifttrace retrain --approve`) and the Airflow approval gate. A dedicated operator web UI is
  not requested by the pitch and is out of MVP scope.

**Acceptance criteria**
1. WHEN an operator requests the latest RCA THEN the system SHALL present the root-cause
   candidate, symptom path, and per-node evidence.
2. WHEN an operator approves a rollback to a named model version THEN serving SHALL load that
   version and the change SHALL be recorded in the audit trail.
3. WHEN retraining is triggered THEN the system SHALL require explicit approval input and record
   who/what approved it.
4. WHEN no approval is given THEN the deployed model version SHALL remain unchanged.

### FR-13 - Streaming drift monitoring
**User story:** As an operator, I want live events monitored in short windows, so that drift is
detected close to when it happens without slowing predictions.

- FR-13.1 `[PITCH]` The system SHALL consume a Kafka / Redpanda event stream.
- FR-13.2 `[PITCH]` The system SHALL process live events in **sliding or tumbling windows**.
- FR-13.3 `[PITCH]` Monitoring SHALL run **asynchronously** so predictions remain low-latency.
- FR-13.4 `[PITCH]` Per-feature PSI / KS-test SHALL be computed per window against the training
  reference baseline.
- FR-13.5 `[DECISION]` Redpanda SHALL be the local broker (Kafka-API compatible, lighter on a
  laptop); the pitch names either as acceptable.
- FR-13.6 `[DECISION]` The event source SHALL be an interface with two implementations: the
  Kafka/Redpanda consumer and a file-replay reader, so drift/RCA logic tests without a broker.
- FR-13.7 `[DECISION]` Default window: tumbling, size and minimum-sample count configurable.

**Acceptance criteria**
1. WHEN events are published to the stream THEN the monitor SHALL consume them into windows per
   the configured window policy.
2. WHEN a window closes THEN per-node drift (FR-9) and RCA (FR-10) SHALL run and a monitoring
   report SHALL be produced.
3. WHEN the monitor is stopped and restarted THEN it SHALL resume without reprocessing already
   committed events.
4. WHEN the monitor is slow or down THEN `POST /predict` SHALL continue to serve predictions.
5. WHEN the file-replay source is used THEN drift/RCA results SHALL be identical to the broker
   path for the same event sequence.

### FR-14 - Explainability
**User story:** As a reviewer, I want individual predictions explained, so that a decision can
be justified and a drift alert connected to model behaviour.

- FR-14.1 `[PITCH]` The system SHALL provide SHAP / LIME explanations for predictions.
- FR-14.2 `[DERIVED]` Explanations SHALL be available for a specific prediction and globally
  (feature importance), reproducible for a given model version.
- FR-14.3 `[DECISION]` SHAP is the primary explainer; LIME is a secondary, comparative explainer,
  since the pitch lists both.
- FR-14.4 `[DECISION]` Explanation generation SHALL be off the hot prediction path - a separate
  endpoint, not part of `POST /predict`.

**Acceptance criteria**
1. WHEN `POST /explain` receives a feature vector THEN the response SHALL contain per-feature
   attributions and the model version used.
2. WHEN a global explanation is requested THEN the system SHALL return ranked feature importance
   for the registered model version.
3. WHEN an explanation is generated THEN it SHALL be attachable to the run/report artifacts.

### FR-15 - Responsible AI and governance
**User story:** As a reviewer, I want fairness, privacy, and governance evidence, so that the
system is defensible and auditable.

- FR-15.1 `[PITCH]` The system SHALL include fairness and privacy checks.
- FR-15.2 `[PITCH]` The project SHALL produce a governance checklist.
- FR-15.3 `[PITCH]` MLflow/DVC SHALL provide the audit trail.
- FR-15.4 `[DECISION]` Fairness SHALL be group metrics (selection rate, TPR/FPR gaps) across a
  declared sensitive attribute of the synthetic dataset, declared explicitly in config and
  never inferred.
- FR-15.5 `[DECISION]` Privacy checks SHALL cover: no PII in logs or artifacts, no raw
  identifiers in the feature store or event payloads, and a documented retention note for
  prediction logs.

**Acceptance criteria**
1. WHEN model evaluation runs THEN fairness metrics SHALL be computed per declared group and
   logged to MLflow.
2. WHEN a privacy check runs over logs and artifacts THEN it SHALL fail if a field on the PII
   deny-list is present.
3. WHEN the governance checklist is reviewed THEN each item SHALL point to a concrete artifact
   or code location as evidence.

### FR-16 - CI/CD automation
**User story:** As a maintainer, I want automated checks and image builds on every change, so
that broken code and broken data contracts never reach deployment.

- FR-16.1 `[PITCH]` GitHub Actions SHALL run unit tests, schema tests, image build, and
  deployment.
- FR-16.2 `[DERIVED]` CI SHALL run without a broker, without Airflow, and without cloud
  credentials.
- FR-16.3 `[DECISION]` "Deployment" in CI/CD for the MVP SHALL mean building/publishing the
  container image plus a Compose-based deploy verification. Cloud deployment is the SageMaker
  stretch goal; no cloud credentials are configured now.
- FR-16.4 `[DECISION]` CI SHALL also run lint, format, and type checks and fail on violations.

**Acceptance criteria**
1. WHEN a commit is pushed or a PR opened THEN CI SHALL run lint, type checks, unit tests, and
   schema tests.
2. WHEN any required check fails THEN the workflow SHALL fail and report which stage failed.
3. WHEN checks pass on the default branch THEN the Docker image SHALL build and be tagged with
   the commit SHA.
4. WHEN the built image starts THEN a smoke test SHALL confirm `GET /health` and one
   `POST /predict` call succeed.

### FR-17 - Monitoring reports and lifecycle evidence
**User story:** As an evaluator, I want each run traceable end to end, so that the project
demonstrates a reproducible lifecycle rather than a one-off notebook result.

- FR-17.1 `[PITCH]` Each run SHALL be reproducible from a versioned dataset, code commit,
  pipeline execution, model artifact, and monitoring report.
- FR-17.2 `[PITCH]` The monitor SHALL produce report artifacts containing root-cause evidence.
- FR-17.3 `[DECISION]` Reports SHALL be machine-readable JSON plus a rendered human-readable
  summary (Markdown/HTML).

**Acceptance criteria**
1. WHEN a monitoring cycle completes THEN a report SHALL be persisted with the window id,
   per-node drift results, RCA verdict, model version, dataset version, and code commit.
2. WHEN a report is opened THEN the five lifecycle-evidence artifacts SHALL be identifiable from
   it.

### FR-18 - Drift-injection harness `[DECISION]`
**User story:** As a developer, I want to inject the failure scenario on demand, so that
detection and RCA can be demonstrated and regression-tested.

Rationale: the pitch defines the monthly->annual `income` failure as the motivating scenario but
does not say how it is produced. A controllable injector is the minimum needed to demonstrate
and test the system, and adds nothing to the product surface beyond scenario generation.

- FR-18.1 The harness SHALL support: no drift (control), the monthly->annual `income` change,
  drift injected at a mid-chain node, and simultaneous independent drift at two roots.
- FR-18.2 Scenarios SHALL be reproducible from a seed and usable by both the file-replay source
  and the stream producer.

**Acceptance criteria**
1. WHEN the monthly->annual scenario runs end to end THEN the system SHALL report `income` as the
   single root-cause candidate with `credit_score` and `risk_score` as symptoms.
2. WHEN the control scenario runs THEN no alert SHALL be emitted.

### FR-19 - Prometheus + Grafana observability `[STRETCH]`
- FR-19.1 `[PITCH]` Prometheus + Grafana SHALL provide metrics, dashboards, alerts, and
  root-cause evidence.
- FR-19.2 `[PITCH]` Alerting via Grafana.
- FR-19.3 `[DERIVED]` The API and monitor SHALL expose a metrics endpoint Prometheus can scrape.
  `[DECISION]` The metrics endpoint may be built during MVP work (cheap, additive), but
  Prometheus and Grafana themselves stay stretch.

**Acceptance criteria (stretch)**
1. WHEN the stretch stack runs THEN Prometheus SHALL scrape API/monitor metrics and Grafana SHALL
   display drift, RCA, and service-health panels.
2. WHEN a root-cause alert fires THEN Grafana alerting SHALL surface it with the evidence.

### FR-20 - AWS SageMaker deployment and monitoring `[STRETCH]`
- FR-20.1 `[PITCH]` AWS SageMaker deployment/monitoring is a stretch goal (Unit VI, Practical 10).
- FR-20.2 `[DERIVED]` MVP code SHALL avoid cloud-specific coupling to keep this path open, but no
  AWS account, region, role, or credential is configured at this stage.

**Acceptance criteria (stretch)**
1. WHEN the stretch path is attempted THEN the container image and model artifact SHALL be
   deployable to a SageMaker endpoint without changes to core drift/RCA logic.

### FR-21 - Future scope `[FUTURE]`
Recorded, not planned for this course project: automatic graph learning; temporal windows +
GNNs. Out of scope for all MVP phases.

---

## 7. Non-functional requirements

| ID | Requirement | Tag | Verification |
| --- | --- | --- | --- |
| NFR-1 | **Reproducibility.** Any run re-creatable from its versioned dataset, code commit, pipeline execution, model artifact, and monitoring report. | `[PITCH]` | Re-run a recorded run; metrics match. |
| NFR-2 | **Prediction latency.** Monitoring asynchronous; prediction latency independent of drift computation. Target: p95 `POST /predict` < 200 ms single record on reference laptop. | `[PITCH]` intent, `[DECISION]` target | Load monitor while measuring API latency. |
| NFR-3 | **Laptop footprint.** MVP runs on a student laptop. Target: full Compose stack < ~6 GB RAM; broker-free profile < ~2 GB. | `[PITCH]` intent, `[DECISION]` budgets | Measure container stats per profile. |
| NFR-4 | **Portability.** Service runs as a portable container with no host-specific paths. | `[PITCH]` | Build and run on clean checkout. |
| NFR-5 | **Observability.** Structured logging, metrics, and alerts for pipeline, API, and monitor. | `[PITCH]` + `[DECISION]` JSON logs | Inspect logs and metrics. |
| NFR-6 | **Maintainability.** Modular code, typed public interfaces, lint/format/type checks in CI. | `[PITCH]` + `[DECISION]` | CI gates. |
| NFR-7 | **Secret hygiene.** No invented credentials, tokens, cloud accounts, or external endpoints. Config via versioned non-secret config + env vars with safe local defaults. | `[DERIVED]` | Secret scan in CI. |
| NFR-8 | **Testability.** Drift detection, graph traversal, RCA pure/deterministic/unit-testable without broker, Airflow, or Docker. | `[DERIVED]` | Unit tests in isolation. |
| NFR-9 | **Privacy.** No PII in logs, artifacts, or event payloads. | `[PITCH]` | Automated privacy check (FR-15). |
| NFR-10 | **Documentation.** Setup, architecture, demo runbook, governance checklist in-repo. | `[PITCH]` + `[DECISION]` | Doc plan review. |
| NFR-11 | **Offline friendliness.** After initial install, the MVP demo runs with no internet. | `[DECISION]` | Run demo with networking disabled. |
| NFR-12 | **Determinism.** Seeds configured and recorded for data generation, splitting, training. | `[DERIVED]` | Repeat-run equality tests. |

## 8. System requirements

`[DECISION]` unless marked otherwise.

**Development host (reference target)**
- OS: Windows 11 + PowerShell (current dev machine); Linux/macOS supported via the same
  containers and scripts.
- CPU 4+ cores; RAM 8 GB minimum, 16 GB comfortable; disk 10 GB free.
- Python 3.11 `[DECISION]` - broad support across MLflow, Airflow, SHAP, Kafka clients.
- Git; Docker Desktop with Compose v2 (required for container/streaming profiles, optional for
  the pure-Python profile).

**Core software** (`[PITCH]` for choice, `[DECISION]` for exact pinning at implementation time):
Git + DVC | MLflow | Apache Airflow | FastAPI + Uvicorn | Docker + Compose | GitHub Actions |
NetworkX | scikit-learn | pandas/numpy | SciPy (KS-test) | SHAP | LIME | Kafka client +
Redpanda broker | pytest. Stretch: Prometheus, Grafana, AWS SageMaker.

**Local ports** `[DECISION]`: API 8000, MLflow 5000, Airflow 8080, Redpanda 9092, webhook stub
9000, Prometheus 9090 (stretch), Grafana 3000 (stretch).

**Network and accounts:** none required for the MVP.

## 9. Project-level acceptance criteria

| ID | Criterion |
| --- | --- |
| AC-1 | `dvc repro` (or documented entry point) reproduces dataset, features, model, and evaluation from a clean checkout. |
| AC-2 | MLflow shows runs with parameters, metrics, artifacts, dataset version, and code commit, plus a registered model with >= 2 versions. |
| AC-3 | The Airflow DAG runs `ingest -> validate -> drift-check -> report / retrain`, and a validation failure demonstrably blocks downstream tasks. |
| AC-4 | The FastAPI service runs in Docker, passes health/readiness checks, serves predictions, and reports its model version. |
| AC-5 | **Core demo:** injecting the monthly->annual `income` change produces exactly one alert naming `income` as root-cause candidate, with `credit_score` and `risk_score` reported as downstream symptoms plus evidence. |
| AC-6 | The control scenario produces no alert. |
| AC-7 | Streaming events are consumed and windowed; monitoring runs asynchronously without degrading prediction latency. |
| AC-8 | SHAP (and LIME) explanations are obtainable for a specific prediction and globally for the model version. |
| AC-9 | Fairness and privacy checks run and are logged; the governance checklist is complete with evidence links. |
| AC-10 | GitHub Actions runs lint, type checks, unit tests, schema tests, builds the image, and smoke-tests the container. |
| AC-11 | A monitoring report ties one drift incident to all five lifecycle-evidence artifacts. |
| AC-12 | An operator-approved rollback switches serving to a previous registered model version and is recorded in the audit trail. |
| AC-13 | Documentation lets a new reader set up and run the demo end to end without verbal guidance. |

Stretch acceptance (not required for MVP): Prometheus/Grafana dashboards and Grafana alerting
(FR-19); SageMaker deployment/monitoring (FR-20).

## 10. Out of scope
- Anything tagged `[FUTURE]`: automatic graph learning, temporal windows + GNNs.
- Causal *discovery* from data. The pitch is explicit that the graph is **declared**.
- Model architecture research or accuracy competition. The deliverable is a real-time,
  reproducible, containerized, monitored ML system, not a model accuracy report.
- A full operator web UI, multi-tenancy, user accounts, or RBAC.
- External data acquisition, paid APIs, or any hosted service.

## 11. Traceability
- Syllabus Units I-VI mapping: `syllabus-mapping.md`.
- Architecture and flows: `design.md`.
- Phased tasks with requirement references: `tasks.md`.
- Every `[DECISION]` is listed with rationale and alternatives in `assumptions-and-decisions.md`.
