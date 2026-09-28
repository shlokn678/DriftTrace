# DriftTrace — Runbook

How to run DriftTrace locally. **No Docker, no message broker, and no cloud accounts
or credentials are required.** The backend runs in a Python virtualenv; the dashboard
runs with Node/Vite.

All commands are run from the repository root. Windows examples use PowerShell; the
`.\.venv\Scripts\python.exe` prefix becomes `./.venv/bin/python` on Linux/macOS.

---

## 0. One-time setup (fresh clone)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This creates `.venv`, installs the project with the `serving,tracking,streaming`
extras, and runs the bootstrap (dataset + schema validation + trained/registered model
+ drift baseline). Cross-platform equivalent:

```powershell
.\.venv\Scripts\python.exe -m drifttrace.bootstrap          # reuse existing state
.\.venv\Scripts\python.exe -m drifttrace.bootstrap --force  # rebuild from scratch
```

Prerequisites: **Python 3.11+**, **Node.js/npm** (dashboard only), **Git**.

---

## 1. Start the backend API

```powershell
.\.venv\Scripts\python.exe -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
```

The API loads the registered model at startup and stays up (reporting non-ready) if no
model is present - re-run the bootstrap in that case.

## 2. Start the dashboard (second terminal)

```powershell
cd frontend
npm install      # first time only
npm run dev
```

Dashboard: http://localhost:5173. API: http://localhost:8000.

## 3. Verify the API

```powershell
curl.exe http://localhost:8000/health        # {"status":"ok"}
curl.exe http://localhost:8000/ready          # {"ready":true,"model_version":"...",...}
curl.exe http://localhost:8000/model-info     # model name + features + loaded
curl.exe http://localhost:8000/models/active  # active model (custom or default loan)
```

Make a prediction:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"income\": 4200.0, \"request_id\": \"demo-1\"}'
```

The response includes `prediction`, `probability`, derived `features`, the serving
`model_version`, and `event_emitted: true`. Prediction events are appended to a local
JSON-lines file (`reports/events.jsonl` by default) - no broker involved.

---

## 4. Model onboarding + activation

```powershell
# Upload one model file (.pkl / .pickle / .joblib); DriftTrace inspects it
curl.exe -F "file=@model.pkl" http://localhost:8000/models/upload

# Activate the returned model id ("Use this model") - /predict now serves it
curl.exe -X POST http://localhost:8000/models/<model_id>/activate

# Restore the default loan model
curl.exe -X POST http://localhost:8000/models/deactivate
```

In the dashboard, use **Add Model** (drag/drop or browse), then **Use this model**.
The default loan model is always the fallback. Reference data and the dependency graph
are reused automatically; only genuinely missing information is requested.

---

## 5. Drift demo -> RCA (real Phase 4 pipeline, broker-free)

The simplest path runs a deterministic scenario in-process through the exact drift ->
RCA -> report -> alert pipeline and writes `reports/latest_rca.json`:

```powershell
# Main pitch demo: monthly -> annual income. Expect income = ROOT CAUSE;
# credit_score + risk_score = SYMPTOMS; exactly one alert.
curl.exe -X POST http://localhost:8000/demo/run-scenario `
  -H "Content-Type: application/json" `
  -d '{\"scenario\": \"income_annual\", \"n\": 300, \"seed\": 7}'
```

Scenarios: `control` (no drift), `income_annual` (income root), `mid_chain`
(credit_score root), `two_roots`. List them: `curl.exe http://localhost:8000/demo/scenarios`.

### CLI equivalent (file replay, no broker)

```powershell
.\.venv\Scripts\python.exe -m drifttrace.cli.main demo-generate --out reports\demo --n 300 --seed 7
.\.venv\Scripts\python.exe -m drifttrace.cli.main replay --file reports\demo\events_income_annual.jsonl `
  --window-size 300 --min-samples 30
```

Add `--webhook-url http://localhost:9000/alert` if you are running the local webhook
stub (`... webhook-stub --port 9000`) and want to see alert delivery.

## 6. Check drift results / RCA / reports / alerts

```powershell
curl.exe http://localhost:8000/rca/latest          # latest persisted RCA report
Get-Content reports\latest_rca.json                 # same report on disk
Get-Content reports\alerts.jsonl                    # emitted alerts (root-cause only)
```

Or open the dashboard: the simple view shows the root cause and affected features; KS/PSI
evidence and the dependency graph are under **View Details**.

---

## 7. Explainability (SHAP / LIME), off the prediction hot path

```powershell
curl.exe -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{\"income\": 4200.0, \"method\": \"shap\"}'
curl.exe -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{\"income\": 4200.0, \"method\": \"lime\"}'
# CLI:
.\.venv\Scripts\python.exe -m drifttrace.cli.main explain --income 4200 --method shap
```

## 8. Governance (fairness + privacy)

```powershell
.\.venv\Scripts\python.exe -m drifttrace.cli.main governance
```

## 9. Operator actions (approval required; audited)

```powershell
# Rejected without approval (exit code 2):
.\.venv\Scripts\python.exe -m drifttrace.cli.main rollback --to-version 1
# Approved (audited):
.\.venv\Scripts\python.exe -m drifttrace.cli.main rollback --to-version 1 --approve --approver alice
# Retrain requires approval too:
.\.venv\Scripts\python.exe -m drifttrace.cli.main retrain --approve --approver alice
Get-Content reports\audit_log.jsonl                 # audit trail
```

## 10. Metrics (Prometheus-compatible text; no Prometheus/Grafana installed)

```powershell
curl.exe http://localhost:8000/metrics
```

---

## MLflow UI (optional)

MLflow uses a local SQLite backend by default (`mlflow.db` in the repo root). To browse
runs and the model registry:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Then open http://localhost:5000.

## Optional: Redpanda streaming (not required)

The normal workflow is broker-free. If you separately run a Redpanda/Kafka broker, set
`DRIFTTRACE_USE_REDPANDA=true` (and `REDPANDA_BROKER`) so the API emits events to a topic,
and run the monitor against it: `... monitor --brokers localhost:9092 --limit 500`.
This is entirely optional and outside the default demo.

## Notes

- The MVP API is unauthenticated and intended for localhost / internal use only
  (FR-7.6). Do not expose it publicly as-is.
- Runtime state (`data/`, `artifacts/`, `reports/`, `mlflow.db`) is git-ignored and fully
  reconstructed by the bootstrap - never copy it between machines.

## Troubleshooting

- **`/ready` is false** - no model loaded; run `python -m drifttrace.bootstrap`.
- **`Python 3.11+ is required`** - install Python 3.11+; `setup.ps1` uses the `py`
  launcher automatically when available.
- **Dashboard shows data as unavailable** - the API is not reachable on port 8000.
- **Reset everything** - `scripts\setup.ps1 -Force` regenerates the dataset + model.
