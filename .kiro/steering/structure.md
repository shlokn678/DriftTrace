---
inclusion: always
---
# DriftTrace — Structure Steering

- `src/drifttrace/` core library, one subpackage per component. Keep it importable and
  unit-testable without a broker. Key packages:
  - `bundle/` model-bundle domain: `loader` (zip/dir), `reference` (profile + baseline),
    `graph_json` (optional graph.json -> DependencyGraph).
  - `adapters/` model-agnostic adapter layer: `base` (interface + standardized event),
    `sklearn_adapter` (the only implemented adapter), `registry` (detect + build + inspect).
  - `drift/` KS/PSI engine + baseline + config. `graph/` NetworkX DAG. `rca/` root-cause
    analysis. `streaming/` prediction-event schema + windowing + monitor + file/Redpanda
    source. `serving/` FastAPI app + onboarding/active-model registry + schemas + config.
  - `explain/` SHAP/LIME. `governance/` audit + privacy + operator (approval) + report.
  - `bootstrap.py` fresh-clone setup (creates runtime dirs only; NO model is trained).
  - `cli/` thin command wrappers (serve, webhook-stub, rollback, retrain, bootstrap).
- `config/` declared thresholds/governance: `drift.yaml`, `governance.yaml`. Feature schema
  and dependency graph are NOT declared here - they come from the uploaded model bundle.
- `scripts/setup.ps1` / `scripts/setup.sh` fresh-clone bootstrap (local-first; no Docker).
- `frontend/` React + TypeScript + Vite dashboard (upload bundle -> activate -> monitor).
- `.github/workflows/` GitHub Actions CI. `tests/{unit,integration,e2e}/` test suite.
- `docs/` architecture, runbook, governance checklist.
- `data/ artifacts/ reports/` runtime outputs (git-ignored except .keep); `artifacts/
  uploaded_models/` is the local model store for onboarded bundles.
- `.kiro/specs/drifttrace/` the source-of-truth spec. Update the spec before large changes.
