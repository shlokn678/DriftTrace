# DriftTrace Operations Dashboard (frontend)

A presentation-quality ML operations dashboard for DriftTrace. It consumes the real
Phase 4 FastAPI backend - no fabricated data. Editorial, restrained visual language
(off-white surfaces, large extrabold headings, JetBrains Mono technical labels, a single
controlled blue accent).

## Stack
- React 18 + TypeScript + Vite
- Lucide icons
- No CSS framework: hand-authored design tokens (`src/styles/tokens.css`)

## Prerequisites
The DriftTrace backend must be running with a trained/registered model. From the repo
root:
```
# ensure a model + baseline + transform params exist
.\.venv\Scripts\python.exe -m drifttrace.cli.main run-pipeline --seed 42 --min-roc-auc 0.6
# start the API
.\.venv\Scripts\python.exe -m uvicorn drifttrace.serving.app:create_app --factory --host 127.0.0.1 --port 8000
```
(Or run the full Docker Compose stack - see ../docs/runbook.md.)

## Develop
```
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api -> http://localhost:8000)
```
Override the backend target with `VITE_API_TARGET` (see `.env.example`).

## Build / typecheck
```
npm run typecheck
npm run build        # outputs dist/
npm run preview
```

## How it connects
All backend calls go through a single typed client (`src/api/client.ts`). Integrated
endpoints:
- `GET /health`, `GET /ready`, `GET /model-info`
- `POST /predict` (unchanged; used implicitly via the backend)
- `GET /rca/latest` (drives the dependency graph + RCA + incident + diagnostics)
- `POST /explain` (SHAP primary / LIME secondary)
- `GET /metrics` (Prometheus-compatible text, parsed client-side)
- `POST /demo/run-scenario` (thin wrapper over the real KS/PSI + graph RCA pipeline)

## Sections
Header - Hero + lifecycle strip - KPI bento - Dependency graph + RCA reasoning -
Incident/alert - SHAP/LIME explanation - Drift diagnostics (KS/PSI) - Operator actions
(demo drivers + approval-gated rollback/retrain) - System/metrics.

## Notes
- Operator rollback/retrain preserve the backend's explicit-approval semantics: the UI
  never triggers automatic rollback or retraining; it surfaces the audited CLI command.
- Unavailable data renders as an explicit "unavailable" state, never as fake values.
