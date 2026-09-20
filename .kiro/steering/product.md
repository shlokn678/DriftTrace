---
inclusion: always
---
# DriftTrace — Product Steering

DriftTrace is an end-to-end MLOps system for automated root-cause analysis of ML drift, built
for course CI3203D. It turns a wall of independent drift alarms into one actionable diagnosis.

Core loop (pitch): detect drift per node (KS-test / PSI) -> read the declared dependency graph
and trace upstream -> flag the earliest drifted node and alert only on that root-cause
candidate -> let operators inspect the evidence and decide to fix data, roll back, or retrain.

Application: a loan-default / churn pipeline with chained features
`income -> credit_score -> risk_score -> prediction API` and a deployed prediction API.

Terminology is fixed by the pitch: root cause, symptom, symptom path, dependency graph,
lifecycle evidence. Preserve it. Do not silently add features. Keep anything the pitch marks as
stretch (Prometheus/Grafana, AWS SageMaker) as stretch, and anything under Future Scope
(automatic graph learning, temporal windows + GNNs) out of the MVP.
