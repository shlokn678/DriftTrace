# DriftTrace — Architecture Reference

This document mirrors the authoritative architecture in
`.kiro/specs/drifttrace/design.md`. When they differ, the spec wins. It will be expanded with
finalized diagrams and interfaces as implementation proceeds.

## Summary
- Core libraries (drift, graph, RCA) are pure, deterministic, and broker-free.
- Airflow, Kafka/Redpanda, FastAPI, and Docker are thin adapters over the core.
- Monitoring is asynchronous; prediction latency never depends on drift computation.
- The dependency graph is declared in `config/graph.yaml` and loaded into NetworkX.
- Alert only on the root-cause candidate; symptoms are recorded as evidence.

See `design.md` for the full component table and the data / lifecycle / drift / RCA / monitoring
/ deployment flows.
