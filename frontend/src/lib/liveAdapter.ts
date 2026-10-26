// Pure live->batch adapter: reshape a persisted live window + live incident into the
// MonitoringReport shape the existing advanced panels (RcaSection / IncidentCard /
// DriftMetrics / DependencyGraph) already consume. This performs NO drift or RCA
// recomputation; every value comes from the existing engine output already persisted in
// LiveWindowResult.per_feature / RCA fields / the live incident.

import type {
  AlertRecord,
  LiveIncident,
  LiveWindowResult,
  MonitoringReport,
  NodeClass,
  NodeDriftResult,
} from "../api/types";

// Display-only threshold constants. DriftMetrics only renders the statistic/p_value/psi/
// current_n values (which are real, from the engine) plus these threshold labels.
const KS_THRESHOLD = 0.05;
const PSI_WARNING = 0.1;
const PSI_DRIFT = 0.2;
const BASELINE_VERSION = "live";

/** Reshape one evaluated live window (+ optional incident) into a MonitoringReport. */
export function liveWindowToReport(
  window: LiveWindowResult,
  incident: LiveIncident | null,
): MonitoringReport {
  const windowId = window.window_id;

  // ---- drift.nodes: straight from per_feature (no recomputation) ----
  const nodes: Record<string, NodeDriftResult> = {};
  for (const [node, pf] of Object.entries(window.per_feature)) {
    nodes[node] = {
      node,
      verdict: pf.verdict,
      ks: {
        node,
        window_id: windowId,
        baseline_version: BASELINE_VERSION,
        current_n: window.event_count,
        baseline_n: 0,
        statistic: pf.ks_statistic,
        p_value: pf.ks_p_value,
        threshold: KS_THRESHOLD,
        verdict: pf.verdict,
      },
      psi: {
        node,
        window_id: windowId,
        baseline_version: BASELINE_VERSION,
        current_n: window.event_count,
        baseline_n: 0,
        psi: pf.psi,
        bins: [],
        psi_warning: PSI_WARNING,
        psi_drift: PSI_DRIFT,
        verdict: pf.verdict,
      },
    };
  }

  // ---- rca: reshape the live RCA fields ----
  const rootCauseCandidates = window.root_cause_candidates.map((node, i) => ({
    node,
    severity: window.drifted_count,
    symptom_path: i === 0 ? window.primary_symptom_path : [],
    evidence: {} as Record<string, unknown>,
  }));

  // ---- classifications: node -> NodeClass (consumed unchanged by DependencyGraph) ----
  const classifications: Record<string, NodeClass> = {};
  for (const [node, pf] of Object.entries(window.per_feature)) {
    let cls: NodeClass;
    if (window.root_cause_candidates.includes(node)) {
      cls = "ROOT_CAUSE";
    } else if (window.symptoms.includes(node)) {
      cls = "SYMPTOM";
    } else if (window.drifted_features.includes(node)) {
      cls = pf.verdict === "WARNING" ? "WARNING" : "SYMPTOM";
    } else if (pf.verdict === "WARNING") {
      cls = "WARNING";
    } else if (pf.verdict === "INSUFFICIENT_DATA") {
      cls = "INSUFFICIENT_DATA";
    } else {
      cls = "STABLE";
    }
    classifications[node] = cls;
  }

  // ---- alerts: pass the engine alerts through; otherwise synthesize ONE display alert
  // ONLY for an OPEN incident (never for a CLOSED/absent incident, so IncidentCard does
  // not claim a closed incident is still active) ----
  let alerts: AlertRecord[] = [...window.alerts];
  if (
    alerts.length === 0 &&
    incident?.status === "OPEN" &&
    window.has_root_cause
  ) {
    alerts = [
      {
        incident_id: incident.incident_id,
        root_cause: incident.root_cause ?? window.primary_root_cause ?? "-",
        model_version: window.model_version ?? "-",
        window_id: windowId,
        symptom_path: incident.symptom_path ?? window.primary_symptom_path,
        evidence: {},
        timestamp: Math.floor(Date.parse(window.evaluated_at) / 1000) || 0,
        delivered: true,
        delivery_status: "live",
      },
    ];
  }

  return {
    report_id: `live-${window.session_id ?? "session"}-${windowId}`,
    window_id: windowId,
    incident_id: incident?.incident_id ?? window.incident_id ?? null,
    model_version: window.model_version ?? "-",
    baseline_version: BASELINE_VERSION,
    dataset_version: null,
    code_commit: null,
    timestamp: window.evaluated_at,
    classifications,
    drift: {
      window_id: windowId,
      baseline_version: BASELINE_VERSION,
      nodes,
      drifted_nodes: window.drifted_features,
    },
    rca: {
      window_id: windowId,
      baseline_version: BASELINE_VERSION,
      has_root_cause: window.has_root_cause,
      root_cause_candidates: rootCauseCandidates,
      symptoms: window.symptoms,
      drifted_nodes: window.drifted_features,
    },
    alerts,
    // Declared dependency order (parents before children) so the Trace Upstream diagram
    // renders arrows in the real direction. Fall back to the RCA chain order
    // (root -> symptoms) when the window predates graph_nodes but has a root + path.
    graph_nodes:
      window.graph_nodes && window.graph_nodes.length > 0
        ? window.graph_nodes
        : window.primary_root_cause
          ? [window.primary_root_cause, ...window.primary_symptom_path]
          : [],
    // Declared edges so the Trace Upstream diagram draws arrows only where a real edge
    // exists (never between unrelated/stable features). Empty for no-graph models.
    graph_edges: window.graph_edges ?? [],
  };
}
