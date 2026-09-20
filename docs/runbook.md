# DriftTrace — Runbook (placeholder)

To be completed during implementation. It will describe, step by step:

1. Setup on a student laptop (Python 3.11, optional Docker Desktop).
2. Reproduce the pipeline (`dvc repro` / CLI) and view runs in MLflow.
3. Start the serving API (`core` Compose profile) and call `/predict`.
4. Run the demo scenarios:
   - Control: no injected drift -> no alert.
   - Core demo: inject the monthly->annual `income` change -> exactly one alert naming
     `income` as root cause, with `credit_score` and `risk_score` as downstream symptoms.
5. Inspect the RCA evidence and, if desired, perform an operator-approved rollback.

No credentials or cloud accounts are required.
