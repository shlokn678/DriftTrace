# DriftTrace — Architecture Reference

This document mirrors the authoritative architecture in
`.kiro/specs/drifttrace/design.md`. When they differ, the spec wins. It will be expanded with
finalized diagrams and interfaces as implementation proceeds.

## Summary
- Core libraries (drift, graph, RCA) are pure, deterministic, and broker-free.
- A model plugs in through a **model adapter** (`src/drifttrace/adapters/`) that emits
  standardized prediction events; the core never sees model internals. The MVP ships one
  adapter (scikit-learn). Flow: `model → adapter → standardized events → core → KS/PSI →
  graph + RCA → alerts/reports/explanations`.
- FastAPI (and optionally Airflow or a separately-run Redpanda broker) are thin
  adapters over the core. The normal local workflow is broker-free and Docker-free.
- Monitoring is asynchronous; prediction latency never depends on drift computation.
- The dependency graph is declared in `config/graph.yaml` and loaded into NetworkX.
- Alert only on the root-cause candidate; symptoms are recorded as evidence.

See `design.md` for the full component table and the data / lifecycle / drift / RCA / monitoring
/ deployment flows.
