import type { NodeClass } from "../api/types";
import { classToStatus, NODE_LABELS } from "../lib/status";
import "./DependencyGraph.css";

// The declared chain: income -> credit_score -> risk_score -> prediction API.
const CHAIN = ["income", "credit_score", "risk_score", "prediction"] as const;

interface Props {
  classifications: Record<string, NodeClass> | null;
}

const STATUS_TEXT: Record<string, string> = {
  root: "ROOT CAUSE",
  drift: "SYMPTOM",
  warning: "WARNING",
  stable: "STABLE",
  insufficient: "NO DATA",
};

export function DependencyGraph({ classifications }: Props) {
  return (
    <div className="depgraph" role="img" aria-label="DriftTrace dependency graph with node drift status">
      {CHAIN.map((node, i) => {
        const cls = classifications?.[node];
        // prediction has no drift classification (model output); treat as neutral output.
        const status = cls ? classToStatus(cls) : node === "prediction" ? "output" : "unknown";
        return (
          <div className="depgraph__row" key={node}>
            <div className={`depnode depnode--${status}`}>
              <div className="depnode__top">
                <span className="depnode__name mono-value">{NODE_LABELS[node]}</span>
                {status === "root" && <span className="depnode__badge">ROOT</span>}
              </div>
              <span className="depnode__status">
                {node === "prediction"
                  ? "OUTPUT"
                  : cls
                    ? STATUS_TEXT[status] ?? "-"
                    : "AWAITING"}
              </span>
            </div>
            {i < CHAIN.length - 1 && (
              <div className="depgraph__edge" aria-hidden>
                <svg width="24" height="40" viewBox="0 0 24 40">
                  <line x1="12" y1="0" x2="12" y2="30" stroke="currentColor" strokeWidth="2" />
                  <path d="M12 40 L6 30 L18 30 Z" fill="currentColor" />
                </svg>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
