# DriftTrace — Runbook

How to run the DriftTrace stack locally. No cloud accounts or credentials are required.

- Phases 0-2 (pipeline, training, DAG, CI) run with the local Python environment.
- Phase 3 (serving + streaming infrastructure) runs via Docker Compose.

## Prerequisites
- Docker Desktop running (engine reachable: `docker info`).
- For the pure-Python path: the project virtualenv at `.venv` with extras installed:
  `python -m pip install -e ".[dev,tracking,serving,streaming]"`.

All commands below are run from the repository root. On Windows use PowerShell.

---

## 1. Build the image
```
docker build -f docker/Dockerfile -t drifttrace:latest .
```

## 2. Start the Phase 3 stack

### core profile (lightweight: init + mlflow + api + webhook-stub)
The `init` service trains and registers the model into a shared volume (using the
existing Phase 1 pipeline and the same model), then the API loads it.
```
docker compose -f docker/docker-compose.yml --profile core up -d
```

### full profile (adds Redpanda + monitor for streaming end-to-end)
```
$env:DRIFTTRACE_USE_REDPANDA = "true"   # PowerShell; API emits events to Redpanda
docker compose -f docker/docker-compose.yml --profile full up -d
```

### stretch profile
Reserved for Prometheus/Grafana (Phase 4 / stretch); not built in Phase 3.

## 3. Verify the API

### health / readiness
```
curl http://localhost:8000/health      # {"status":"ok"}
curl http://localhost:8000/ready        # {"ready":true,"model_version":"1",...}
curl http://localhost:8000/model-info   # model name + features + loaded
```

### make a prediction
```
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"income": 4200.0, "request_id": "demo-1"}'
```
The response includes `prediction`, `probability`, derived `features`, the serving
`model_version`, and `event_emitted: true`.

### verify a prediction event was emitted
- core profile (file sink):
  ```
  docker exec drifttrace-api-1 sh -c "tail -n 1 /store/reports/events.jsonl"
  ```
- full profile (Redpanda): events are produced to the `drifttrace.predictions` topic;
  the monitor consumes them (next step).

## 4. Deterministic demo / replay (normal + simulated drift)

The replay path proves the infrastructure carries BOTH normal and simulated-drift
events through: sample events -> event source -> monitor/window -> webhook stub.
Simulated drift is demo input only; NO drift detection is performed (that is Phase 4).

Generate the deterministic event files and replay them (pure-Python, broker-free):
```
python -m drifttrace.cli.main demo-generate --out reports/demo --n 200 --seed 7
python -m drifttrace.cli.main replay --file reports/demo/events_normal.jsonl `
  --webhook-url http://localhost:9000/alert --window-size 50 --min-samples 10
python -m drifttrace.cli.main replay --file reports/demo/events_drift.jsonl `
  --webhook-url http://localhost:9000/alert --window-size 50 --min-samples 10
```

## 5. Verify the monitor processed events (full profile, via Redpanda)
After making predictions with the full profile, run the monitor as a one-shot to
consume from Redpanda, window the events, and notify the webhook stub:
```
docker compose -f docker/docker-compose.yml --profile full run --rm monitor `
  monitor --window-size 50 --min-samples 10 --limit 500
```
The output reports `windows_processed` and `notifications_sent`.

## 6. Verify the webhook stub received events
```
curl http://localhost:9000/count       # {"count": N}
curl http://localhost:9000/received     # full list of received window summaries
```

## 7. View logs
```
docker compose -f docker/docker-compose.yml --profile full logs           # all
docker compose -f docker/docker-compose.yml --profile full logs api        # one service
docker logs drifttrace-monitor-1
```

## 8. Stop the stack
```
docker compose -f docker/docker-compose.yml --profile core --profile full down
```

## 9. Clean up containers + volumes
```
docker compose -f docker/docker-compose.yml --profile core --profile full down -v
```
The `-v` flag also removes the `model-store` volume (the trained model + MLflow store);
the next `up` re-runs `init` to retrain deterministically.

---

## MLflow UI
With the stack up, the MLflow tracking/registry UI is at http://localhost:5000.

## Notes
- The MVP API is unauthenticated and intended for localhost / the internal Compose
  network only (FR-7.6). Do not expose it publicly as-is.
- The MLflow server image is pinned to match the client version so the shared SQLite
  store migrates cleanly.


---

# Phase 4 — Drift detection, RCA, alerting (Operate)

Phase 4 turns the Phase 3 infrastructure into the real DriftTrace intelligence:
`prediction events -> window -> KS+PSI drift -> per-node verdicts -> declared graph ->
RCA -> root-cause-only alert -> monitoring report -> operator decision`.

Prometheus/Grafana are deferred (stretch); only a `/metrics` endpoint is provided.

## Start the Phase 4 stack
```
docker build -f docker/Dockerfile -t drifttrace:latest .
$env:DRIFTTRACE_USE_REDPANDA = "true"     # full profile emits events to Redpanda
docker compose -f docker/docker-compose.yml --profile full up -d
```
`init` trains + registers the model into the shared volume; `api`, `redpanda`,
`webhook-stub`, and `monitor` then start.

## Generate the deterministic scenarios (inside the api container)
```
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 `
  python -m drifttrace.cli.main demo-generate --out /store/reports/demo --n 300 --seed 7
```
This writes `events_control.jsonl`, `events_income_annual.jsonl`,
`events_mid_chain.jsonl`, `events_two_roots.jsonl`.

## Run monitoring on a scenario (drift -> RCA -> report -> alert)
```
# Control (expect: no drift, no root cause, no alert)
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main `
  replay --file /store/reports/demo/events_control.jsonl `
  --webhook-url http://webhook-stub:9000/alert --window-size 300 --min-samples 30

# Main pitch demo: monthly -> annual income (expect: income = root cause; one alert)
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main `
  replay --file /store/reports/demo/events_income_annual.jsonl `
  --webhook-url http://webhook-stub:9000/alert --window-size 300 --min-samples 30

# Mid-chain drift (expect: credit_score = root cause)
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main `
  replay --file /store/reports/demo/events_mid_chain.jsonl `
  --webhook-url http://webhook-stub:9000/alert --window-size 300 --min-samples 30
```

## Check drift results / RCA / reports / alerts
```
curl http://localhost:8000/rca/latest                 # latest persisted RCA report
docker exec drifttrace-api-1 sh -c "ls /store/reports/*.md | head -1 | xargs cat"  # a report
docker exec drifttrace-api-1 sh -c "cat /store/reports/alerts.jsonl"               # alerts
curl http://localhost:9000/count                       # webhook stub receipts
curl http://localhost:9000/received                    # received alert payloads
```

## Explainability (SHAP / LIME), off the prediction hot path
```
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{"income": 4200.0, "method": "shap"}'
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{"income": 4200.0, "method": "lime"}'
# or via the CLI:
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main explain --income 4200 --method shap
```

## Governance (fairness + privacy)
```
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main governance
```

## Operator actions (approval required; audited)
```
# Rejected without approval (exit code 2):
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main rollback --to-version 1
# Approved (audited):
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main rollback --to-version 1 --approve --approver alice
# Retrain requires approval too:
docker exec -e DRIFTTRACE_ROOT=/store drifttrace-api-1 python -m drifttrace.cli.main retrain --approve --approver alice
docker exec drifttrace-api-1 sh -c "cat /store/reports/audit_log.jsonl"   # audit trail
```

## Metrics (Prometheus-compatible; no Prometheus/Grafana installed)
```
curl http://localhost:8000/metrics
```

## Stop / clean up
```
docker compose -f docker/docker-compose.yml --profile core --profile full down
docker compose -f docker/docker-compose.yml --profile core --profile full down -v   # also removes model-store
```
