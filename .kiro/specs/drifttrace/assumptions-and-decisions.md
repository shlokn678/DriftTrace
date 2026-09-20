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
