# DriftTrace — Implementation Plan (Tasks)

Phases follow the pitch roadmap (Reproduce -> Automate -> Deploy -> Operate -> Cloud + Future)
and are ordered by dependency. Each task references requirements from `requirements.md`.

> ## SUPERSESSION NOTE (Phase 5 — final model-agnostic overhaul)
>
> Phases 0–4 were implemented for the loan-default application, then Phase 5 replaced that
> application with a **model-agnostic** system: no built-in model; users upload a bundle
> (`model.pkl` + `reference.csv` + optional `graph.json`); a scikit-learn adapter isolates
> the model; drift/RCA are generic over any features; the dependency graph is optional; no
> Docker; local-first. The loan generator/transform/training pipeline, the Airflow DAG, and
> the `config/graph.yaml`/`schema.yaml` files were removed. Older loan-specific tasks below
> are historical; the current behavior is described in `docs/architecture.md`,
> `docs/runbook.md`, and `assumptions-and-decisions.md` (D-52..D-56).

Legend: `[ ]` not started · MVP unless marked `[STRETCH]`/`[FUTURE]`.

## Phase 0 — Project foundation (prerequisite) — DONE (commit 269d237)
- [x] 0.1 Initialize Git repo and Python 3.11 project (pyproject, package skeleton). _(NFR-6)_ — used Python 3.13.5 (D-29; 3.11 not installed).
- [x] 0.2 Add lint/format/type/test tooling config (ruff, black, mypy, pytest). _(NFR-6, FR-16.4)_
- [x] 0.3 Add `.env.example`, config file stubs (`graph.yaml`, `schema.yaml`, `drift.yaml`, `governance.yaml`). _(design §13, NFR-7)_
- [x] 0.4 Author README skeleton + docs index. _(NFR-10)_

## Phase 1 — Reproduce (Data + Track) `[PITCH Phase 1: DVC + MLflow + validation]` — DONE (commit bfd6c0f)
- [x] 1.1 Deterministic seeded data generator with chained features. _(FR-1.4, NFR-12)_
- [x] 1.2 Declared schema + validator + validation report. _(FR-2)_
- [x] 1.3 Initialize DVC with local remote; track dataset/model artifacts; `dvc.yaml` stages. _(FR-1)_ — `dvc repro` runs generate->train end to end.
- [x] 1.4 Feature transform module (shared by train/serve). _(FR-3)_
- [x] 1.5 Graph loader (`graph.yaml` -> NetworkX) with validation + traversal API. _(FR-8)_
- [x] 1.6 Feature/graph consistency check. _(FR-3.2)_
- [x] 1.7 Training + evaluation (deterministic split, metrics, gate, artifact). _(FR-4)_
- [x] 1.8 MLflow tracking + model registry integration; record lifecycle evidence. _(FR-5, FR-17)_ — SQLite backend (D-6); cloudpickle serialization.
- [x] 1.9 Versioned drift baseline per model version. _(FR-9.2/9.3)_

## Phase 2 — Automate (DAG + CI) `[PITCH Phase 2: Airflow DAG + CI tests]` — DONE
- [x] 2.1 CLI wrapping library functions (`run-pipeline`, etc.). _(FR-6.4)_ — added `ingest`, `validate`, `drift-check`, `report`, `retrain --approve`, `run-dag`.
- [x] 2.2 Airflow DAG `ingest -> validate -> drift-check -> report / retrain` with approval gate. _(FR-6)_ — thin wrapper over `orchestration/stages.py`; retrain gated on approval (D-34). Airflow runtime is POSIX-only, so validated at code level on Windows (D-36); runs on Linux/WSL/Docker.
- [x] 2.3 GitHub Actions: lint, format, type, unit, schema tests. _(FR-16.1/16.4)_ — `.github/workflows/ci.yml` (Python 3.11 + 3.13).
- [x] 2.4 CI runs broker-free / Airflow-free / credential-free. _(FR-16.2)_ — installs only `.[dev,tracking]`; no broker/Airflow/Docker/secrets (D-37).

## Phase 3 — Deploy (API + Docker + streaming) `[PITCH Phase 3: FastAPI + Docker + Kafka]` — DONE
- [x] 3.1 FastAPI service: `/health`, `/ready`, `/predict`, `/model-info`; load model by version; request logging. _(FR-7)_ — `serving/app.py`; loads registered model via MLflow registry; logs each request.
- [x] 3.2 Dockerfile + Compose profiles (`core`, `full`). _(NFR-3/4, design §11)_ — single image, entrypoint roles api/webhook/monitor/cli; profiles core/full (+stretch reserved). Built and run on Docker 29.8.0.
- [x] 3.3 Event source interface: file-replay + Redpanda consumer. _(FR-13.6)_ — `streaming/source.py`; core stays broker-free (confluent-kafka lazy import).
- [x] 3.4 Windowing + async streaming monitor process. _(FR-13.1/13.2/13.3)_ — `streaming/window.py` + `streaming/monitor.py`; tumbling windows; NO drift algorithms (Phase 4).
- [x] 3.5 Prediction event emission from API (fire-and-forget). _(FR-7.2, NFR-2)_ — `PredictionEvent` emitted to file (core) or Redpanda (full); failures never break serving.
- [x] 3.6 Redpanda + monitor + webhook-stub services in Compose. _(FR-11.5, design §11)_ — verified e2e: predict -> Redpanda -> monitor -> webhook stub (count observed).
- [x] 3.7 CI image build + Compose smoke test (`/health` + one `/predict`). _(FR-16.3, AC-10)_ — `.github/workflows/ci.yml` `docker` job; smoke path verified locally.
- Deterministic demo/replay (REQUIRED): `streaming/demo.py` + `replay` CLI; normal + simulated-drift event files carried through source -> monitor/window -> webhook. Simulated drift is demo input only; NOT drift detection (Phase 4).

## Phase 4 — Operate (Drift + RCA + Alerts) `[PITCH Phase 4: Prometheus/Grafana + RCA]` — DONE
- [x] 4.1 Per-node drift detection (KS + PSI, verdicts). _(FR-9)_ — `drift/ks.py`, `drift/psi.py`, `drift/engine.py`; verdicts STABLE/WARNING/DRIFT/INSUFFICIENT_DATA; documented KS+PSI combine rule (PSI-gated to avoid KS large-sample false positives).
- [x] 4.2 RCA engine (upstream trace, earliest node, symptom path, evidence). _(FR-10)_ — `rca/engine.py`; co-equal roots on branched graphs.
- [x] 4.3 Root-cause-only alerting with cool-down + webhook + local persistence. _(FR-11)_ — `alerting/alerter.py`; incident ids; delivery status; webhook failure never crashes.
- [x] 4.4 Monitoring report (JSON + summary) with lifecycle evidence. _(FR-17)_ — `governance/report.py` (JSON + Markdown; latest_rca.json).
- [x] 4.5 `/rca/latest` endpoint. _(FR-7.5, FR-12.1)_ — serving reads latest_rca.json.
- [x] 4.6 Operator actions: `rollback --to-version`, `retrain --approve`; audit trail. _(FR-12)_ — `governance/operator.py` + `governance/audit.py`; approval enforced (exit 2 without), audited.
- [x] 4.7 Drift-injection harness (control, monthly->annual income, mid-chain, two-roots). _(FR-18)_ — `streaming/demo.py`; values only, detector decides. Verified e2e in the container.
- [x] 4.8 Explainability: SHAP (primary) + LIME; `/explain` + global importance. _(FR-14)_ — `explain/explainer.py`; off hot path; caveat included; in image.
- [x] 4.9 Responsible AI: fairness metrics, privacy/PII check, governance evidence. _(FR-15)_ — `governance/fairness.py`, `governance/privacy.py`; declared `group` attribute.
- [x] 4.10 `/metrics` endpoint (Prometheus-compatible text). _(FR-19.3)_ — `serving/metrics.py`; counters for predictions/events/windows/drift/alerts/etc.
- [ ] 4.11 `[STRETCH]` Prometheus + Grafana services, dashboards, Grafana alerting. _(FR-19)_ — DEFERRED as stretch (only the /metrics interface exists; no Prometheus/Grafana installed).

## Phase 5 — Cloud `[PITCH Phase 5: SageMaker; stretch]`
- [ ] 5.1 `[STRETCH]` Keep image/model cloud-portable; document SageMaker deploy/monitor path (no account configured). _(FR-20)_

## Future scope `[FUTURE]`
- [ ] F.1 `[FUTURE]` Automatic graph learning.
- [ ] F.2 `[FUTURE]` Temporal windows + GNNs.

---

## Testing strategy (D)

**Unit** _(NFR-8)_
- Drift math: KS/PSI on known distributions incl. monthly->annual income (FR-9 AC-3).
- Graph: build/validate; reject cycles/undefined/orphans; ancestor queries (FR-8).
- RCA: earliest-node selection, symptom classification, multi-root case (FR-10).
- Schema validation pass/fail with named columns (FR-2).
- Feature transform train==serve equality (FR-3 AC-3).
- Determinism: repeat-run equality for generator and training (NFR-12).
- Alerting: exactly-one-alert, cool-down, webhook-failure resilience (FR-11).

**Integration**
- Pipeline stages end to end via CLI without Airflow (FR-6.4).
- MLflow logging + registry + load-by-version (FR-5).
- API endpoints incl. 422 handling and readiness (FR-7).
- File-replay vs broker parity (FR-13.6 AC-5).
- Validation failure blocks downstream in the DAG (FR-6 AC-2, AC-3).

**End-to-end** _(scenario = pitch demo)_
- E2E-1 control: no injected drift -> no alert (AC-6).
- E2E-2 core demo: monthly->annual income -> exactly one alert naming `income`, symptoms
  `credit_score`/`risk_score` with evidence (AC-5).
- E2E-3 mid-chain drift -> correct single candidate.
- E2E-4 async: predict latency stable under monitor load (AC-7, NFR-2).
- E2E-5 report ties one incident to all five lifecycle-evidence artifacts (AC-11).
- E2E-6 operator-approved rollback switches version + audit entry (AC-12).

**CI gates:** lint, format, type, unit, schema on every push/PR; image build + Compose smoke on
default branch (FR-16, AC-10). Coverage target and thresholds set in Phase 0/2.

## Local development strategy (E) `[DECISION]`
- `core` profile (api + mlflow + webhook-stub) runs on a student laptop, broker-free, < ~2 GB
  (NFR-3); pure-Python path runs pipeline + unit tests with no Docker (FR-6.4, NFR-8).
- `full` profile adds Redpanda + monitor for the streaming demo.
- File-replay source enables the whole drift/RCA demo without a broker.
- Offline after first install (NFR-11). Seeds recorded for reproducibility (NFR-12).
- `scripts/` holds setup + demo runner (documented in the runbook).

## Docker / container strategy (F) `[PITCH + DECISION]`
- Single project image (`docker/Dockerfile`); entrypoint selects api/monitor/cli.
- `docker/docker-compose.yml` with profiles `core` / `full` / `stretch` (design §11).
- Health checks on `api`; portable, no host paths (NFR-4).
- Stretch profile wires Prometheus + Grafana only when requested.

## CI/CD strategy (G) `[PITCH: GitHub Actions]`
- Workflow: checkout -> install -> lint/format -> type -> unit + schema tests -> build image
  (tag = commit SHA) -> Compose smoke test. Secret scan (NFR-7). No cloud creds; SageMaker
  deploy stays stretch (FR-16.3, FR-20).

## Documentation plan (H) `[PITCH governance + DECISION]`
- `README.md` — overview, quickstart, profiles.
- `docs/architecture.md` — mirrors design.md diagrams/flows.
- `docs/runbook.md` — run the pitch demo end to end (control + monthly->annual income).
- `docs/governance-checklist.md` — fairness/privacy/audit items with evidence links (FR-15).
- `.kiro/specs/drifttrace/syllabus-mapping.md` — Units I-VI mapping.
- `.kiro/specs/drifttrace/assumptions-and-decisions.md` — all `[DECISION]` entries.

## Dependency ordering summary
Phase 0 -> 1 -> 2 -> 3 -> 4; Phase 5 and Future are stretch/out-of-scope. Within Phase 4, drift
(4.1) precedes RCA (4.2) precedes alerting (4.3); reports (4.4) and endpoints (4.5) follow.
