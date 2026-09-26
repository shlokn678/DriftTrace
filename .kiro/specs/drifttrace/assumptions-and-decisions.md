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
