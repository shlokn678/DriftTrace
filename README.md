# DriftTrace

**An End-to-End MLOps System for Automated Root-Cause Analysis of ML Drift**
Course CI3203D · Computer Science & Engineering (Artificial Intelligence)

> Status: **planning / specification stage.** No application code has been implemented yet.
> The specification is the source of truth and lives in `.kiro/specs/drifttrace/`.

## What it does
A model can be accurate in training and still fail in production. DriftTrace turns a wall of
independent drift alarms into one actionable diagnosis:

1. Detect drift per node using KS-test / PSI.
2. Read the declared dependency graph and trace upstream.
3. Flag the earliest drifted node; log and alert only on the root-cause candidate.
4. Let operators inspect the evidence and decide whether to fix data, roll back, or retrain.

Application: a loan-default / churn pipeline with chained features
`income -> credit_score -> risk_score -> prediction API`.

## MVP stack
FastAPI + Docker Compose + MLflow + GitHub Actions, with Git + DVC, Apache Airflow, NetworkX,
scikit-learn, SciPy, SHAP/LIME, and a Kafka-API broker (Redpanda). **Stretch:** Prometheus /
Grafana and AWS SageMaker. **Future:** automatic graph learning, temporal windows + GNNs.

## Specification (start here)
- `.kiro/specs/drifttrace/requirements.md` — functional / non-functional / system requirements + acceptance criteria
- `.kiro/specs/drifttrace/design.md` — architecture, component responsibilities, data & lifecycle flows
- `.kiro/specs/drifttrace/tasks.md` — phased implementation plan, testing, Docker, CI/CD, docs plans
- `.kiro/specs/drifttrace/syllabus-mapping.md` — CI3203D Units I-VI mapping
- `.kiro/specs/drifttrace/assumptions-and-decisions.md` — every implementation decision + open questions

## Docs
- `docs/architecture.md` — architecture reference (mirrors the spec)
- `docs/runbook.md` — how to run the demo (to be filled during implementation)
- `docs/governance-checklist.md` — responsible-AI evidence checklist

## Repository layout
See `.kiro/steering/structure.md`. Directories are scaffolded; implementation begins at
Phase 0 in `tasks.md`.

## Quickstart (planned)
Implementation not started. Planned local dev uses a broker-free `core` Docker Compose profile
plus a pure-Python path for the pipeline and unit tests. Details will land in `docs/runbook.md`.

## Notes
- Not yet connected to GitHub.
- No cloud accounts, credentials, external datasets, or paid APIs are required or configured.
- The MVP prediction API is unauthenticated and intended for localhost / internal use only.
