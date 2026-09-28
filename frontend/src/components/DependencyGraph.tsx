import type { NodeClass } from "../api/types";
import { classToStatus, nodeLabel } from "../lib/status";
import "./DependencyGraph.css";

interface Props {
  /** Node -> classification from the latest report (dynamic; any feature set). */
  classifications: Record<string, NodeClass> | null;
  /** Whether a dependency graph was provided for the active model. */
  dependenciesAvailable?: boolean;
}

const STATUS_TEXT: Record<string, string> = {
  root: "ROOT CAUSE",
  drift: "SYMPTOM",
  warning: "WARNING",
  stable: "STABLE",
  insufficient: "NO DATA",
};

/**
 * Renders the monitored nodes and their drift status straight from the backend
 * classifications. Feature names are dynamic - nothing is hard-coded. The output node
 * ("prediction") is shown last as the model output.
 */
export function DependencyGraph({ classifications, dependenciesAvailable = true }: Props) {
  const entries = classifications ? Object.keys(classifications) : [];
  // Put the model output node last; keep the rest in their reported order.
  const nodes = entries
    .filter((n) => n !== "prediction")
    .concat(entries.includes("prediction") ? ["prediction"] : []);

  if (nodes.length === 0) {
    return (
      <div className="depgraph" role="img" aria-label="No monitored nodes yet">
        <div className="depnode depnode--unknown">
          <span className="depnode__status">Awaiting a monitoring window</span>
        </div>
      </div>
    );
  }

  return (
    <div className="depgraph" role="img" aria-label="Monitored nodes and drift status">
      {nodes.map((node, i) => {
        const cls = classifications?.[node];
        const status = cls
          ? classToStatus(cls)
          : node === "prediction"
            ? "output"
            : "unknown";
        return (
          <div className="depgraph__row" key={node}>
            <div className={`depnode depnode--${status}`}>
              <div className="depnode__top">
                <span className="depnode__name mono-value">{nodeLabel(node)}</span>
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
            {i < nodes.length - 1 && dependenciesAvailable && (
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
