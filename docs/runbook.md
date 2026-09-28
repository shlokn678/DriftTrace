# DriftTrace — Runbook

How to run DriftTrace locally. **No Docker, no message broker, and no cloud accounts or
credentials are required.** The backend runs in a Python virtualenv; the dashboard runs
with Node/Vite. There is **no built-in model** — you upload one.

All commands run from the repository root. Windows examples use PowerShell; the
`.\.venv\Scripts\python.exe` prefix becomes `./.venv/bin/python` on Linux/macOS.

---

## 0. One-time setup (fresh clone)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This creates `.venv`, installs the project (`serving,tracking,streaming` extras), and
prepares runtime directories. It does **not** create, train, or activate any model.
Cross-platform equivalent:

```powershell
.\.venv\Scripts\python.exe -m drifttrace.bootstrap
```

Prerequisites: **Python 3.11+**, **Node.js/npm** (dashboard only), **Git**.

---

## 1. Start the backend API

```powershell
.\.venv\Scripts\python.exe -m drifttrace.cli.main serve --host 127.0.0.1 --port 8000
```

The API starts with **no active model** and reports non-ready until you activate one.

## 2. Start the dashboard (second terminal)

```powershell
cd frontend
npm install      # first time only
npm run dev
```

Dashboard: http://localhost:5173. API: http://localhost:8000.

## 3. Verify the API

```powershell
curl.exe http://localhost:8000/health         # {"status":"ok"}
curl.exe http://localhost:8000/ready           # {"ready":false,...} until a model is active
curl.exe http://localhost:8000/models/active   # {"active":false,...} on a fresh install
```

---

## 4. Build a model bundle

A bundle is a `.zip` of `model.pkl` + `reference.csv` (+ optional `graph.json`). Any
scikit-learn estimator or `Pipeline` works. Example (adapt to your own model):

```python
import io, json, zipfile, cloudpickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
ref = pd.DataFrame({
    "feature_a": rng.normal(40, 10, 400),
    "feature_b": rng.normal(50000, 12000, 400),
    "feature_c": rng.normal(5, 2, 400),
})
y = (ref["feature_b"] + ref["feature_a"] * 100 > 55000).astype(int)
model = LogisticRegression(max_iter=200).fit(ref, y)

with zipfile.ZipFile("my_model.drift.zip", "w") as zf:
    buf = io.BytesIO(); cloudpickle.dump(model, buf)
    zf.writestr("model.pkl", buf.getvalue())
    zf.writestr("reference.csv", ref.to_csv(index=False))
    # optional graph for RCA:
    zf.writestr("graph.json", json.dumps(
        {"edges": [["feature_a", "feature_b"], ["feature_b", "feature_c"]]}))
```

Omit `graph.json` to see the no-graph behavior (drift only, no root-cause tracing).

## 5. Upload and activate the model

```powershell
# Upload the bundle .zip and inspect it
curl.exe -F "file=@my_model.drift.zip" http://localhost:8000/models/upload
# -> returns model_id, task, features, reference/dependencies availability

# Activate it ("Use this model") - /predict now serves it
curl.exe -X POST http://localhost:8000/models/<model_id>/activate

# Deactivate (there is no fallback; monitoring stops)
curl.exe -X POST http://localhost:8000/models/deactivate
```

In the dashboard: drop the bundle onto **Add Model**, review the inspection, then click
**Use this model**.

## 6. Predict

Use the active model's own feature names:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"features\": {\"feature_a\": 40, \"feature_b\": 60000, \"feature_c\": 5}}'
```

The response has `prediction` (class, or `null` for regressors), `probability`, `output`
(numeric), the `features` used, and `event_emitted: true`. Events are appended to
`reports/events.jsonl` (no broker).

## 7. Run a drift test -> RCA (real pipeline)

Perturbs a sample of the active model's reference data and runs the real KS/PSI + RCA
pipeline, writing `reports/latest_rca.json`:

```powershell
# Drift test (shift the reference distribution)
curl.exe -X POST http://localhost:8000/demo/run-drift-test `
  -H "Content-Type: application/json" `
  -d '{\"intensity\": 2.0, \"n\": 300, \"seed\": 7}'

# Baseline test (no shift -> expect no drift)
curl.exe -X POST http://localhost:8000/demo/run-drift-test `
  -H "Content-Type: application/json" `
  -d '{\"intensity\": 0.0, \"n\": 300, \"seed\": 7}'
```

With a graph, the earliest drifted feature is the root cause and downstream features are
symptoms. Without a graph, features are independent — drift is reported but the origin is
undetermined. In the dashboard, use **Run Drift Test** under **View Details**.

## 8. Check the diagnosis

```powershell
curl.exe http://localhost:8000/rca/latest      # latest persisted monitoring/RCA report
Get-Content reports\latest_rca.json             # same report on disk
Get-Content reports\alerts.jsonl                # emitted alerts (root-cause only)
```

## 9. Explainability (SHAP / LIME), off the prediction hot path

```powershell
curl.exe -X POST http://localhost:8000/explain -H "Content-Type: application/json" `
  -d '{\"features\": {\"feature_a\": 40, \"feature_b\": 60000, \"feature_c\": 5}, \"method\": \"shap\"}'
```

Use `"method": "lime"` for the secondary explainer. Attribution is not proof of a causal
root cause — it is kept separate from RCA.

## 10. Operator actions (approval required; audited)

DriftTrace never rolls back or retrains automatically. Because models are uploaded (not
trained here), `retrain` records the decision only — you retrain outside DriftTrace and
upload the new bundle.

```powershell
# Rejected without approval (exit code 2):
.\.venv\Scripts\python.exe -m drifttrace.cli.main rollback --to-model <id>
# Approved (audited):
.\.venv\Scripts\python.exe -m drifttrace.cli.main rollback --to-model <id> --approve --approver alice
.\.venv\Scripts\python.exe -m drifttrace.cli.main retrain --approve --approver alice
Get-Content reports\audit_log.jsonl
```

## 11. Metrics (Prometheus-compatible text; no Prometheus/Grafana installed)

```powershell
curl.exe http://localhost:8000/metrics
```

---

## Model switching

Uploading and activating a different bundle switches the active model: the feature list,
reference baseline, and dependency graph all change, and the previous model's monitoring
report is cleared so results never mix.

## Optional: Redpanda streaming (not required)

The normal workflow is broker-free. If you separately run a Redpanda/Kafka broker, set
`DRIFTTRACE_USE_REDPANDA=true` (and `REDPANDA_BROKER`) so the API emits events to a topic.
This is entirely optional and outside the default workflow.

## Notes

- The API is unauthenticated and intended for localhost / internal use only. Do not expose
  it publicly as-is.
- Runtime state (`data/`, `artifacts/`, `reports/`) is git-ignored; uploaded bundles live
  under `artifacts/uploaded_models/`. Never copy runtime state between machines — re-upload
  the bundle instead.

## Troubleshooting

- **`/ready` is false / `/predict` returns 503** — no active model; upload and activate a
  bundle.
- **`Python 3.11+ is required`** — install Python 3.11+; `setup.ps1` uses the `py` launcher
  automatically when available.
- **Dashboard shows data as unavailable** — the API is not reachable on port 8000.
- **Upload rejected (422)** — the file is not a valid bundle: it must be a `.zip` containing
  at least `model.pkl` and `reference.csv`.
