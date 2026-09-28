# DriftTrace — Architecture Reference

DriftTrace is a model-agnostic ML monitoring and root-cause diagnosis system. A user uploads
a model bundle; DriftTrace inspects it, the user activates it, and incoming predictions feed
drift detection and (when a dependency graph is provided) root-cause analysis.

## End-to-end flow

```
USER MODEL BUNDLE (model.pkl + reference.csv + optional graph.json)
    -> MODEL ADAPTER (scikit-learn)          # standardized model interface
    -> STANDARDIZED PREDICTION EVENT          # feature vector + output, no model internals
    -> DRIFTTRACE CORE
    -> KS / PSI DRIFT DETECTION               # reference vs recent window, per feature
    -> OPTIONAL DEPENDENCY GRAPH + RCA        # trace upstream to the earliest drifted node
    -> INCIDENT / ALERT / MONITORING REPORT
    -> HUMAN OPERATOR                          # decide: fix data, roll back, retrain
```

## Components

- **Model bundle** (`src/drifttrace/bundle/`): loads a `.zip` (or directory) containing
  `model.pkl` (required), `reference.csv` (required), and `graph.json` (optional). Profiles the
  reference into a feature schema + per-feature distributions, and parses the optional graph.
- **Model adapter** (`src/drifttrace/adapters/`): the small, framework-agnostic interface
  (`ModelAdapter`) the core talks to. The sklearn adapter wraps any estimator or `Pipeline`,
  detects the feature names and task (classification/regression), and returns a
  `PredictionResult` (prediction / probability / numeric output / feature vector). The core
  never sees the sklearn object. scikit-learn is the only implemented adapter; XGBoost /
  LightGBM / ONNX / PyTorch are possible future adapters (not implemented).
- **Reference baseline** (`bundle/reference.py`, `drift/baseline.py`): each model has its own
  baseline built from its reference data — numeric features (KS/PSI) and categorical features
  (frequency PSI), plus the model's reference output distribution for output drift.
- **Drift engine** (`src/drifttrace/drift/`): per-feature KS + PSI vs the baseline. A node is
  DRIFT if PSI >= 0.20, or KS is significant AND PSI >= 0.10; KS-significant-but-small-PSI is
  WARNING; otherwise STABLE. Insufficient data is reported, never called STABLE.
- **Dependency graph + RCA** (`src/drifttrace/graph/`, `src/drifttrace/rca/`): the optional
  graph is a DAG over feature nodes. RCA flags each drifted node with no drifted ancestor as a
  root-cause candidate and marks downstream drift as symptoms, producing one focused incident.
  Without a graph, drift is still detected but root-cause tracing is unavailable.
- **Serving** (`src/drifttrace/serving/`): FastAPI app + an in-memory onboarding/active-model
  registry. `/predict` serves the active model with a generic feature-vector payload and emits
  a standardized event (fire-and-forget). A generic drift test perturbs the active model's
  reference data and runs it through the real pipeline.
- **Streaming** (`src/drifttrace/streaming/`): the `PredictionEvent` schema, tumbling-window
  monitor, and a file event sink/source (broker-free). An optional Redpanda source/sink exists
  but is not used by the normal workflow.
- **Explainability** (`src/drifttrace/explain/`): SHAP (primary) / LIME (secondary) over the
  active model's features. Kept conceptually separate from RCA — attribution is not proof of a
  causal root cause.
- **Governance** (`src/drifttrace/governance/`): PII privacy deny-list checks, an audit trail,
  and operator actions (rollback / retrain) that require explicit approval and are audited.
  DriftTrace never retrains or rolls back automatically.

## Principles

- Broker-free and Docker-free by default; direct local execution is the supported runtime.
- Monitoring is asynchronous; prediction latency never depends on drift computation.
- Alert only on the root-cause candidate; symptoms are recorded as evidence.
- No built-in model; a fresh install starts with no active model.
- Paths resolve from the project root (or `DRIFTTRACE_*` env overrides).

See `.kiro/specs/drifttrace/design.md` for the detailed component/interface reference.
