# DriftTrace — Assumptions & Implementation Decisions

Every `[DECISION]` in the spec is listed here with rationale and alternatives. These are the
places the pitch left unspecified. They are engineering choices, not pitch content, and are open
to change. No credentials, cloud accounts, datasets, or external services are invented.

| ID | Decision | Rationale | Alternative(s) | Reversible? |
| --- | --- | --- | --- | --- |
| D-1 (FR-1.3) | DVC remote = local directory in workspace | No cloud account required; laptop-friendly | S3/GDrive remote later | Yes |
| D-2 (FR-1.4) | Synthetic seeded dataset with chained features | Pitch names no dataset; reproducible; no external download | Use a public finance dataset if permitted | Yes |
| D-3 (FR-2.4) | Lightweight schema module | Small footprint vs heavyweight DQ framework | Great Expectations / Pandera | Yes |
| D-4 (FR-4.3) | scikit-learn (LogReg + gradient-boosted tree) | Pitch requires no specific model; fast on laptop | XGBoost/LightGBM | Yes |
| D-5 (FR-4.4) | Metrics: ROC-AUC primary + PR-AUC/acc/prec/rec/F1 | Standard for imbalanced credit risk | Different metric emphasis | Yes |
| D-6 (FR-5.4) | MLflow local file/SQLite backend | No hosted server/account | Remote MLflow server | Yes |
| D-7 (FR-3.4/8.4) | Declared graph in YAML -> NetworkX | Human-editable single source of truth; matches "declared graph" | JSON/py module | Yes |
| D-8 (FR-6.4) | DAG tasks wrap CLI-callable library fns | Enables CI + laptop runs without Airflow | Airflow-only tasks | Yes |
| D-9 (FR-7.5) | Endpoint set incl. /explain and /rca/latest | Surfaces explainability + RCA per FR-14/FR-10 | Fewer endpoints | Yes |
| D-10 (FR-7.6) | MVP API unauthenticated, localhost/internal only | Pitch requests no auth; SECURITY LIMITATION, not public-safe | Add API key/OAuth | Yes |
| D-11 (FR-9.4) | KS p<0.05; PSI >=0.2 drift / 0.1-0.2 warn | Common defaults; per-node configurable | Tune per node | Yes |
| D-12 (FR-9.5) | Min window sample size -> "insufficient data" | Avoids false drift on tiny windows | Fixed alerting regardless | Yes |
| D-13 (FR-10.5) | Multiple no-ancestor roots reported co-equal, ranked by severity | Honest vs guessing one winner | Heuristic single pick | Yes |
| D-14 (FR-11.2) | MVP alert channel = webhook + logs | Pitch says "Grafana / webhook"; Grafana is stretch | Grafana alerting now | Yes |
| D-15 (FR-11.4) | Cool-down suppresses repeat alerts | Prevents one alert per window for a persistent fault | No suppression | Yes |
| D-16 (FR-11.5) | Local webhook stub service | No third-party notifier/account | Slack/Teams/Email later | Yes |
| D-17 (FR-13.5) | Redpanda as local broker | Kafka-API compatible, lighter on laptop | Apache Kafka | Yes |
| D-18 (FR-13.6) | Event source interface + file-replay | Enables broker-free tests/CI and demo | Broker-only | Yes |
| D-19 (FR-13.7) | Tumbling windows by default | Simplest; sliding configurable | Sliding default | Yes |
| D-20 (FR-14.3) | SHAP primary, LIME secondary | Pitch lists both; SHAP suits the model types | LIME primary | Yes |
| D-21 (FR-14.4) | /explain off the hot path | Protects prediction latency (NFR-2) | Inline explanations | Yes |
| D-22 (FR-15.4) | Fairness via declared sensitive attribute | Synthetic data; never infer sensitive attrs | Multiple attributes | Yes |
| D-23 (FR-15.5) | Privacy = PII deny-list + retention note | Concrete, testable privacy check | Broader privacy tooling | Yes |
| D-24 (FR-16.3) | CI "deploy" = image build + Compose smoke | No cloud creds now; SageMaker stays stretch | Cloud deploy in CI | Yes |
| D-25 (FR-17.3) | Reports = JSON + human-readable summary | Usable in CI and demo | JSON only | Yes |
| D-26 (FR-19.3) | /metrics endpoint may exist in MVP; Prom/Grafana stretch | Cheap + additive; keeps stretch open | Defer entirely | Yes |
| D-27 (System) | Python 3.11 | Broad support across MLflow/Airflow/SHAP/Kafka | 3.10 / 3.12 | Yes |
| D-28 (Deploy) | Compose profiles core/full/stretch | Fits laptop RAM budgets (NFR-3) | Single monolithic stack | Yes |

## Open questions to confirm with the course instructor / user
1. Is a synthetic dataset acceptable, or is a specific finance dataset required? (affects D-2)
2. Is any repository/registry (GitHub, container registry) available for CI publish, or local
   only for now? (affects FR-16.3) — project is not yet connected to GitHub.
3. Should Prometheus/Grafana be attempted within the MVP timeline or kept strictly stretch?
4. Is the SageMaker stretch expected to be actually deployed, or documented as a path only?

## Explicitly NOT assumed / NOT invented
- No cloud accounts, regions, roles, or credentials.
- No third-party notification, tracking, or data services.
- No external dataset download or paid API.
- No authentication provider.


## Decisions taken during implementation (Phase 0)

| ID | Decision | Rationale | Reversible? |
| --- | --- | --- | --- |
| D-29 (supersedes D-27) | Use Python 3.13.5 for the dev environment, not 3.11. | This laptop has 3.10, 3.13, and 3.14 installed but NOT 3.11, and no py-launcher 3.11 runtime. 3.13 is a stable released CPython with full wheel coverage for the core stack (numpy, pandas, scipy, scikit-learn, networkx, pydantic, fastapi, mlflow, pytest). 3.14 was rejected as too new for reliable wheels. `pyproject.toml` keeps `requires-python = ">=3.11"` so 3.11 remains supported where available. | Yes - install 3.11 later and recreate the venv. |
| D-30 | Airflow is an optional extra (`[orchestration]`), not a core dependency. | Keeps the core library, drift/RCA, and serving installable and testable on Python 3.13 without waiting on Airflow's slower support for new Python versions. Matches the spec principle that Airflow is a thin adapter and the pipeline is CLI-callable without it (FR-6.4). Airflow support on 3.13 will be validated in Phase 2. | Yes |
| D-31 | Docker is not installed on this laptop; Docker/Compose tasks (Phase 3) will be authored as files and validated by lint/config where possible, with runtime execution deferred to a machine that has Docker. | `docker` is not on PATH here. The spec already provides a broker-free pure-Python path (FR-6.4, NFR-8) so Phases 0-2 and the core of Phase 4 do not need Docker. Flagged so the container runtime step is not silently skipped. | Yes |
| D-32 | Files are written as UTF-8 without BOM. | Python `tomllib` and many tools reject a BOM at the start of `pyproject.toml`. Enforced repo-wide. | n/a |


## Decisions taken during implementation (Phase 2)

| ID | Decision | Rationale | Reversible? |
| --- | --- | --- | --- |
| D-33 | Orchestration stages live in `src/drifttrace/orchestration/stages.py` as pure, CLI-callable functions; the Airflow DAG (`pipelines/airflow/dags/drifttrace_dag.py`) is a thin wrapper over them. | Directly satisfies FR-6.4 (every DAG task wraps a library function callable from the CLI) and keeps drift/RCA/graph logic broker-free and testable without Airflow (NFR-8, steering). | Yes |
| D-34 | The DAG task graph is `ingest -> validate -> drift_check -> [report, retrain]`. The human-approval gate lives inside the `retrain` task: it retrains only when the DAG run is triggered with conf `{"approved": true}` (Airflow) or `--approve` (CLI). Report always runs. | FR-6.1, FR-6.2, FR-6 AC-3: retraining is never silent; the report branch always runs; validation failure raises and stops downstream (FR-6 AC-2). | Yes |
| D-35 | Interim drift-check detector = relative population-mean shift vs the stored baseline (continuous nodes). Real computation, deterministic, behind a single call site. | Full KS/PSI per-node detection is Phase 4 (FR-9). Phase 2 needs a genuine (non-stub) drift signal to drive the DAG branching; the detector is isolated so Phase 4 swaps in KS/PSI without touching the DAG or CLI. | Yes - replaced in Phase 4. |
| D-36 (runtime limitation) | **Apache Airflow 3.3.2 installs on Python 3.13, but its runtime does not run on native Windows** (Airflow warns it is POSIX-only; its imports use `os.register_at_fork` and the `standard` provider fails to import on Windows). The DAG is therefore validated at the Python/code level in the main venv (import-safe + task-callable + gate tests) and Airflow install is confirmed in an isolated `.venv-airflow`. Full DagBag/scheduler execution requires Linux/WSL2/containers, as Airflow itself documents. | Honors the resume constraint: do not block Phase 2 on Airflow, keep the Airflow layer as the thin adapter, document the limitation, keep the core pipeline fully functional without Airflow (runnable via `drifttrace run-dag`). No Python version change was needed. | Yes - runs as-is on a POSIX host / WSL / the Phase 3 Docker Airflow service. |
| D-37 | CI (GitHub Actions) installs only `.[dev,tracking]` and runs lint/format/type/unit+schema tests on Python 3.11 and 3.13. No Airflow, no broker, no Docker, no secrets. | FR-16.1/16.2/16.4, NFR-7: CI is broker-free, Airflow-free, and credential-free. Airflow's runtime is POSIX-only and unnecessary for CI, which exercises the same stage functions directly. | Yes |


## Decisions taken during implementation (Phase 3)

| ID | Decision | Rationale | Reversible? |
| --- | --- | --- | --- |
| D-38 | Single project image with an entrypoint that selects a role (`api`/`webhook`/`monitor`/`cli`). | One build, many roles; matches design §11 ("single image; entrypoint selects api/monitor/cli"). Smaller maintenance surface than per-role images. | Yes |
| D-39 | Compose `init` one-shot service trains + registers the model into a shared named volume (`model-store`) using the EXISTING Phase 1 pipeline, then `api`/`mlflow` read it. | Makes the containerized demo deterministic and self-contained on any OS: MLflow artifact paths become Linux-native inside the volume, avoiding host Windows path leakage. Uses the existing model/lifecycle; no second model invented. | Yes |
| D-40 | The API loads the model directly from the shared SQLite MLflow store (`sqlite:////store/mlflow.db`), not via the MLflow HTTP server. The `mlflow` service is the UI/registry browser only. | Model loading is deterministic and does not depend on the UI server being healthy; keeps the prediction path simple and offline-friendly. | Yes |
| D-41 | The MLflow server image is pinned to match the client MLflow version (v3.16.1). | A newer client SQLite schema cannot be migrated by an older server image (observed: alembic "Can't locate revision" with v2.17.2). Matching versions fixes the UI startup. | Yes |
| D-42 | Phase 3 monitor uses a NON-statistical default window handler (counts + per-feature means + sources) posted to the webhook stub. | Phase 3 is infrastructure only. KS/PSI drift detection and RCA are Phase 4 and are deliberately NOT implemented; the handler is a single swappable seam for Phase 4. | Yes - Phase 4 swaps the handler. |
| D-43 | Simulated-drift demo data = the pitch's monthly->annual income (x12), generated deterministically. Used ONLY as replay input to exercise the pipeline. | Provides a visibly different event stream to prove the infrastructure carries both normal and drift events; no drift *detection* is performed on it (Phase 3 boundary). | Yes |
| D-44 | Serving event sink is file-based by default (core profile) and Redpanda-based when `DRIFTTRACE_USE_REDPANDA=true` (full profile). Event emission is fire-and-forget and never fails a prediction. | Keeps the core profile broker-free and lightweight (NFR-3) while the full profile exercises the real Kafka-API path (FR-13). Protects prediction latency (FR-7.2, NFR-2). | Yes |
| D-45 | The isolated `.venv-airflow` (Phase 2) and Docker are the supported paths for Airflow; native Windows Airflow runtime remains unsupported and unchanged in Phase 3. | Preserves Phase 2 behavior; Phase 3 adds Docker as the supported container path per the resume instruction. | n/a |

Runtime limitation observed and resolved in Phase 3: the MLflow UI server initially failed
(schema migration) due to a client/server version mismatch; fixed by pinning the server image to
v3.16.1 (D-41). No unresolved Phase 3 runtime failures remain.


## Decisions taken during implementation (Phase 4)

| ID | Decision | Rationale | Reversible? |
| --- | --- | --- | --- |
| D-46 | KS+PSI combine rule is PSI-gated: a node is DRIFT only if PSI >= psi_drift, or KS is significant AND PSI >= psi_warning; KS-significant-but-small-PSI is WARNING (watch, no alert). | The two-sample KS test becomes over-sensitive at large baseline sample sizes and rejects on trivial, practically meaningless differences, producing false DRIFT on control data. PSI is an effect-size measure robust to sample size. Gating on PSI keeps both signals (FR-9.1) while making the control scenario correctly report no drift. Documented in `drift/engine.py`; not a hidden score. | Yes - thresholds/rule are in one place. |
| D-47 | "Two independent roots" is demonstrated on a genuinely branched graph (unit test), while the linear production chain income->credit_score->risk_score correctly yields a single earliest root. | In a linear chain every downstream drifted node has the earliest node as a transitive ancestor, so it is a SYMPTOM by the RCA rule (FR-10.4), never a co-equal root. Co-equal roots require independent source branches; the RCA engine supports them generically and is tested on a branched graph. The two_roots injection scenario therefore honestly shows income as the single root with risk_score as a symptom. | Yes |
| D-48 | The monitor's per-window Phase 4 processing (drift->RCA->report->alert) runs in the monitor process/CLI, separate from the API process. Metrics are per-process counters. | Keeps monitoring asynchronous and off the prediction hot path (FR-13.3, NFR-2). The API exposes its own /metrics; the monitor increments the shared registry within its process. A future Prometheus scrape would target each process. | Yes |
| D-49 | Rollback records the operator decision + target version in the audit trail and expects serving to read the pinned version via `DRIFTTRACE_MODEL_VERSION`; it does not mutate the registry in place. | Deterministic and testable; no automatic/implicit registry changes. Serving already supports pinning a version via env (Phase 3). | Yes |
| D-50 | The Docker image installs the `explain` extra (SHAP/LIME) so `/explain` works in the container. | `/explain` is a required Phase 4 endpoint; it must actually run in the deployed container, not only on the host. | Yes |
| D-51 (deferred) | Prometheus and Grafana are NOT installed or configured. Only the Prometheus-compatible `/metrics` text endpoint exists. | Explicitly deferred as stretch per the Phase 4 instructions and FR-19. The interface is future-compatible so a later phase can add scraping/dashboards without code changes. | Yes |

## Phase 5 (extension) — local-first, Docker removal

| ID | Decision | Rationale | Reversible? |
| --- | --- | --- | --- |
| D-52 (supersedes D-24, D-28, D-31, D-37 Docker clause, D-38, D-50) | Docker is removed entirely. The primary (and only supported) runtime is direct local execution: a Python virtualenv + FastAPI for the backend and Node/Vite for the dashboard. `docker/` (Dockerfile, docker-compose.yml, entrypoint.sh), the CI `docker` job, and Docker-specific env/paths (`/store`, `DRIFTTRACE_ROOT=/store`, service-DNS hostnames) are deleted. | Phase 5 requires a teammate to clone from GitHub and run on another machine without Docker. The core was already broker-free and pure-Python (FR-6.4, NFR-8), so nothing in drift/RCA/serving depended on containers. Removing Docker eliminates the container-only paths and the machine-specific `/store` volume. | Yes - a container image could be re-added later over the same CLI. |
| D-53 | Runtime state is reconstructed deterministically by a bootstrap (`python -m drifttrace.bootstrap`, `scripts/setup.ps1`, `scripts/setup.sh`) from tracked files only: runtime dirs, synthetic dataset, schema validation, trained+registered model, drift baseline, transform params. No runtime artifacts are copied between machines. | Fresh-clone reproducibility (NFR-4, NFR-12) without Docker's shared volume. The CI `bootstrap` job runs it and a `scripts/smoke_api.py` API smoke test in place of the old Compose smoke test. | Yes |
| D-54 | Redpanda/Kafka is optional and separately-run, never required for the normal local/demo workflow. The default event path is the local `FileSink`/`FileReplaySource`; the demo scenario runs fully in-process. `confluent_kafka` stays a lazy import behind the `streaming` extra. | Keeps the broker-independent core (FR-13.6) while honoring "no Docker / no required infrastructure". The event-source abstraction and file-replay path are retained. | Yes |
| D-55 | Paths resolve from the project root or `DRIFTTRACE_*` env overrides (`DRIFTTRACE_ROOT`, `DRIFTTRACE_DATA_DIR`, `DRIFTTRACE_ARTIFACTS_DIR`, `DRIFTTRACE_REPORTS_DIR`, `DRIFTTRACE_MODEL_STORE`); no machine-specific absolute paths anywhere. `Paths` gains `model_store` (default `artifacts/uploaded_models`). | Portability (NFR-4): the repo works when copied to any directory. Centralized in `config.py::get_paths()`. | Yes |
| D-56 | An uploaded, supported, ready-to-monitor model can become the *active* model via `POST /models/{id}/activate` ("Use this model"); `/predict` then serves it through the same adapter boundary. The trained loan model remains the default and fallback (`POST /models/deactivate` restores it). The registry is in-memory (MVP), not a database. | Completes the onboarding flow the pitch requires while keeping "registered" (uploaded/inspected) distinct from "active" (serving). No database is introduced (out of MVP scope). | Yes |
