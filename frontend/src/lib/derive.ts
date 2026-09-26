import type { Metrics, ModelInfoResponse, MonitoringReport } from "../api/types";
import type { StatusKind } from "./status";

export interface MetricsBundle {
  modelVersion: string | null;
  driftStatus: { text: string; status: StatusKind };
  rootCause: string | null;
  activeAlerts: number;
  driftedCount: number;
  windowsProcessed: number | string;
  predictionRequests: number | string;
  predictionEvents: number | string;
}

/** Derive KPI values ONLY from real backend data. Missing values render as "-". */
export function deriveBundle(
  report: MonitoringReport | null,
  metrics: Metrics | null,
  modelInfo: ModelInfoResponse | null,
): MetricsBundle {
  const drifted = report?.drift.drifted_nodes ?? [];
  const hasRoot = report?.rca.has_root_cause ?? false;
  const rootCause = report?.rca.root_cause_candidates?.[0]?.node ?? null;

  let driftStatus: { text: string; status: StatusKind };
  if (!report) {
    driftStatus = { text: "Unknown", status: "insufficient" };
  } else if (drifted.length === 0) {
    driftStatus = { text: "Stable", status: "stable" };
  } else if (hasRoot) {
    driftStatus = { text: "Drift", status: "drift" };
  } else {
    driftStatus = { text: "Warning", status: "warning" };
  }

  const m = (name: string): number | string =>
    metrics && name in metrics ? metrics[name] : "-";

  return {
    modelVersion: modelInfo?.model_version ?? report?.model_version ?? null,
    driftStatus,
    rootCause,
    activeAlerts: report?.alerts.length ?? 0,
    driftedCount: drifted.length,
    windowsProcessed: m("drifttrace_windows_processed_total"),
    predictionRequests: m("drifttrace_prediction_requests_total"),
    predictionEvents: m("drifttrace_prediction_events_total"),
  };
}
