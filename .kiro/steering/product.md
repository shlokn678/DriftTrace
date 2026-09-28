---
inclusion: always
---
# DriftTrace — Product Steering

DriftTrace is an intelligent, model-agnostic ML monitoring and root-cause diagnosis system,
built for course CI3203D. It turns a wall of independent drift alarms into one actionable
diagnosis.

Core loop: detect drift per feature (KS-test / PSI) against a model's reference data -> when a
dependency graph is provided, trace upstream to flag the earliest drifted node as the
root-cause candidate and mark downstream drift as symptoms -> turn related drift into one
focused incident/alert -> let a human operator inspect the evidence and decide what to do.

Product goal: detect drift -> trace where it likely originated -> explain the diagnosis ->
focus the alert -> let the human decide the response.

Model-agnostic contract: the user uploads a bundle containing `model.pkl` (required),
`reference.csv` (required) and `graph.json` (optional). There is NO built-in, default, or
fallback model; a fresh install starts with no model registered and no active model. Any
supported scikit-learn estimator or Pipeline works (classification or regression); the
reference data defines the feature schema and drift baseline; the optional graph enables
dependency-based RCA.

Terminology: root cause, symptom, symptom path, dependency graph, reference data, prediction
event, incident. Preserve it. Do not silently add features. Fairness needs a declared
sensitive attribute (never inferred). Keep stretch items (Prometheus/Grafana, AWS SageMaker)
as stretch, and Future Scope (automatic graph learning, temporal windows + GNNs, causal
discovery) out of the MVP. scikit-learn is the only implemented model adapter; other
frameworks are documented as future work only.
