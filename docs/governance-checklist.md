# DriftTrace — Governance Checklist (placeholder)

Each item will be completed with an evidence link (artifact or code location) during Phase 4.

## Explainability (FR-14)
- [ ] SHAP local explanation available per prediction (`/explain`).
- [ ] SHAP/LIME global feature importance per model version.

## Fairness (FR-15)
- [ ] Sensitive attribute declared in `config/governance.yaml` (never inferred).
- [ ] Group metrics (selection rate, TPR/FPR gaps) computed and logged to MLflow.

## Privacy (FR-15, NFR-9)
- [ ] No PII in logs, artifacts, or event payloads (PII deny-list check passes).
- [ ] Data-retention note for prediction logs documented.

## Audit trail (FR-15.3)
- [ ] MLflow/DVC record links dataset version, code commit, model version, and reports.
- [ ] Rollback/retraining actions recorded with approver.

## Reproducibility (NFR-1, NFR-12)
- [ ] Seeds recorded; a run reproduces from the five lifecycle-evidence artifacts.
