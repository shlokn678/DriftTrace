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
root (run the one-time bootstrap first if you haven't):
```
# ensure runtime state exists (dataset + model + baseline)
.\.venv\Scripts\python.exe -m drifttrace.bootstrap
# start the API
.\.venv\Scripts\python.exe -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
```
See ../docs/runbook.md for the full local-first workflow.

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
