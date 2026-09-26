import type { NodeClass, Verdict } from "../api/types";

export type StatusKind = "stable" | "warning" | "drift" | "root" | "insufficient";

/** Map a node classification to a status kind used for pill/color styling. */
export function classToStatus(cls: NodeClass): StatusKind {
  switch (cls) {
    case "ROOT_CAUSE":
      return "root";
    case "SYMPTOM":
      return "drift";
    case "WARNING":
      return "warning";
    case "INSUFFICIENT_DATA":
      return "insufficient";
    default:
      return "stable";
  }
}

export function verdictToStatus(v: Verdict): StatusKind {
  switch (v) {
    case "DRIFT":
      return "drift";
    case "WARNING":
      return "warning";
    case "INSUFFICIENT_DATA":
      return "insufficient";
    default:
      return "stable";
  }
}

export const STATUS_LABEL: Record<StatusKind, string> = {
  stable: "Stable",
  warning: "Warning",
  drift: "Drift",
  root: "Root cause",
  insufficient: "Insufficient data",
};

export const NODE_LABELS: Record<string, string> = {
  income: "income",
  credit_score: "credit_score",
  risk_score: "risk_score",
  prediction: "prediction API",
};

export function formatNumber(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  if (Math.abs(n) > 0 && Math.abs(n) < 1e-3) return n.toExponential(2);
  return n.toFixed(digits);
}

export function formatInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Math.round(n).toLocaleString();
}
