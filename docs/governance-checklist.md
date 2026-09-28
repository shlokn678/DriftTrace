# DriftTrace — Governance Checklist

Governance is model-agnostic and evidence-based. Items that depend on operator-declared
configuration (e.g. a sensitive attribute) are marked "when provided".

## Explainability
- [x] SHAP local explanation per prediction over the active model's features (`/explain`).
- [x] LIME as a secondary local explainer.
- [x] Explanations carry a caveat: attribution is not proof of a causal root cause.

## Fairness (when provided)
- [ ] A sensitive attribute is DECLARED in `config/governance.yaml` (never inferred). No
      default is set, so fairness analysis is unavailable unless the operator declares an
      attribute that exists in their model's features.

## Privacy (NFR-9)
- [x] No PII in logs, artifacts, or event payloads — the PII deny-list check
      (`governance/privacy.py`) flags forbidden field names.
- [x] Data-retention note for prediction logs documented in `config/governance.yaml`.

## Audit trail
- [x] Operator rollback/retrain actions are recorded in an append-only audit trail with the
      approver (`governance/operator.py`, `governance/audit.py`).
- [x] DriftTrace never retrains or rolls back automatically; both require explicit approval.
      Because models are uploaded (not trained by DriftTrace), automated retraining is not
      available — the retrain action records the decision only.

## Reproducibility
- [x] Deterministic drift test (seeded) reproduces the same monitoring outcome.
- [x] A fresh clone reconstructs runtime directories via the bootstrap; no runtime artifacts
      are copied between machines. No built-in model is created.
