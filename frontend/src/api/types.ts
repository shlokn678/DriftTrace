// Typed shapes for the real DriftTrace backend responses.
// Derived from src/drifttrace/serving/schemas.py and the Phase 4 report/RCA/explain dicts.

export type Verdict = "STABLE" | "WARNING" | "DRIFT" | "INSUFFICIENT_DATA";
export type NodeClass =
  | "ROOT_CAUSE"
  | "SYMPTOM"
  | "WARNING"
  | "STABLE"
  | "INSUFFICIENT_DATA";

export interface HealthResponse {
  status: string;
}

export interface ReadyResponse {
  ready: boolean;
  model_version: string | null;
  detail: string;
}

export interface ModelInfoResponse {
  model_version: string | null;
  model_name: string;
  features: string[];
  loaded: boolean;
}

export type FeatureValue = number | string;

export interface PredictResponse {
  request_id: string | null;
  model_version: string | null;
  prediction: number | null;
  probability: number | null;
  output: number | null;
  features: Record<string, FeatureValue>;
  event_emitted: boolean;
}

// ---- KS / PSI evidence ----
export interface KSEvidence {
  node: string;
  window_id: string;
  baseline_version: string;
  current_n: number;
  baseline_n: number;
  statistic: number | null;
  p_value: number | null;
  threshold: number;
  verdict: Verdict;
}

export interface PSIBin {
  label: string;
  baseline_prop: number;
  current_prop: number;
  contribution: number;
}

export interface PSIEvidence {
  node: string;
  window_id: string;
  baseline_version: string;
  current_n: number;
  baseline_n: number;
  psi: number | null;
  bins: PSIBin[];
  psi_warning: number;
  psi_drift: number;
  verdict: Verdict;
}

export interface NodeDriftResult {
  node: string;
  verdict: Verdict;
  ks: KSEvidence | null;
  psi: PSIEvidence | null;
}

export interface DriftReport {
  window_id: string;
  baseline_version: string;
  nodes: Record<string, NodeDriftResult>;
  drifted_nodes: string[];
}

export interface RootCauseCandidate {
  node: string;
  severity: number;
  symptom_path: string[];
  evidence: Record<string, unknown>;
}

export interface RCAResult {
  window_id: string;
  baseline_version: string;
  has_root_cause: boolean;
  root_cause_candidates: RootCauseCandidate[];
  symptoms: string[];
  drifted_nodes: string[];
}

export interface AlertRecord {
  incident_id: string;
  root_cause: string;
  model_version: string;
  window_id: string;
  symptom_path: string[];
  evidence: Record<string, unknown>;
  timestamp: number;
  delivered: boolean;
  delivery_status: string;
}

export interface MonitoringReport {
  report_id: string;
  window_id: string;
  incident_id: string | null;
  model_version: string;
  baseline_version: string;
  dataset_version: string | null;
  code_commit: string | null;
  timestamp: string;
  classifications: Record<string, NodeClass>;
  drift: DriftReport;
  rca: RCAResult;
  alerts: AlertRecord[];
}

export interface RcaLatestResponse {
  available: boolean;
  detail?: string;
  report?: MonitoringReport;
  dependencies_available?: boolean;
}

// ---- Explain ----
export interface Attribution {
  feature: string;
  value: number;
  attribution: number;
}

export interface ExplanationFactor {
  feature: string;
  label: string;
  direction: "toward_prediction" | "away_from_prediction";
  strength: "strong" | "moderate" | "small";
  value: number;
}

export interface Interpretation {
  method: string;
  scope_label: string;
  prediction_label: string;
  summary: string;
  supporting_factors: ExplanationFactor[];
  opposing_factors: ExplanationFactor[];
  other_note: string | null;
  caveat: string;
}

export interface Explanation {
  method: "shap" | "lime";
  scope: "local" | "global";
  model_version: string | null;
  features: string[];
  attributions: Attribution[];
  base_value: number | null;
  caveat: string;
  interpretation: Interpretation | null;
}

// ---- Drift test (generic; perturbs the active model's reference data) ----
export interface DriftTestOutcome {
  window_id: string;
  drifted_nodes: string[];
  has_root_cause: boolean;
  root_cause_candidates: string[];
  symptoms: string[];
  report_id: string;
  alerts: AlertRecord[];
  report_paths: Record<string, string>;
}

export interface DriftTestResponse {
  intensity: number;
  feature: string | null;
  windows: number;
  dependencies_available: boolean;
  outcome: DriftTestOutcome | null;
}

// ---- Metrics ----
export type Metrics = Record<string, number>;

// ---- Phase 5: model onboarding ----
export interface UploadModelResponse {
  model_id: string | null;
  supported: boolean;
  framework: string | null;
  name: string | null;
  task: string | null;
  n_features: number | null;
  supports_proba: boolean;
  features: string[];
  reference_available: boolean;
  dependencies_available: boolean;
  reference_rows: number | null;
  missing: string[];
  ready_to_monitor: boolean;
  message: string | null;
}

export interface ModelStatusResponse {
  model_id: string;
  supported: boolean;
  framework: string | null;
  name: string | null;
  task: string | null;
  n_features: number | null;
  reference_available: boolean;
  dependencies_available: boolean;
  ready_to_monitor: boolean;
  active: boolean;
  message: string | null;
}

export interface ActiveModelResponse {
  active: boolean;
  model_id: string | null;
  name: string | null;
  framework: string | null;
  task: string | null;
  model_version: string | null;
  features: string[];
  dependencies_available: boolean;
  supports_proba: boolean;
  loaded: boolean;
}
