import type { NodeClass } from "../api/types";
import { classToStatus, nodeLabel } from "../lib/status";
import "./DependencyGraph.css";

interface Props {
  /** Node -> classification from the latest report (dynamic; any feature set). */
  classifications: Record<string, NodeClass> | null;
  /** Whether a dependency graph was provided for the active model. */
  dependenciesAvailable?: boolean;
  /**
   * Declared dependency order (parents before children, i.e. the graph's topological
   * order) from the backend. When provided, nodes on the graph chain are rendered in
   * this order so the arrows follow the real dependency direction (root -> symptoms),
   * matching the RCA result. Nodes not in this list keep their reported order and are
   * appended after. Nothing is hard-coded: the order is the declared graph's own.
   */
  order?: string[] | null;
  /**
   * Declared dependency edges as [parent, child] pairs from the backend graph. An arrow
   * between two consecutively-rendered nodes is drawn ONLY when that exact pair is a
   * declared edge; unrelated/stable neighbours get a gap, never a fabricated arrow.
   * When null/empty, no edges are known and no connectors are drawn.
   */
  edges?: string[][] | null;
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
export function DependencyGraph({
  classifications,
  dependenciesAvailable = true,
  order = null,
  edges = null,
}: Props) {
  const entries = classifications ? Object.keys(classifications) : [];

  // Declared edge set as a fast "parent\u0000child" lookup. An arrow between two
  // consecutively-rendered nodes is drawn only when this set contains that pair, so
  // unrelated/stable neighbours are never falsely connected and edges are never inferred
  // from display order.
  const edgeSet = new Set((edges ?? []).map(([p, c]) => `${p}\u0000${c}`));
  const hasEdge = (parent: string, child: string) => edgeSet.has(`${parent}\u0000${child}`);

  // Order the monitored nodes by the declared dependency graph when we have it, so the
  // arrows follow the real parent -> child direction (root cause upstream, symptoms
  // downstream) instead of the backend dict's feature order. Nodes present in the
  // report but not on the declared chain keep their reported order and come after the
  // chain. The model output node is always rendered last.
  const ordered: string[] = [];
  const seen = new Set<string>();
  const push = (n: string) => {
    if (n !== "prediction" && entries.includes(n) && !seen.has(n)) {
      ordered.push(n);
      seen.add(n);
    }
  };
  if (order && order.length > 0) {
    for (const n of order) push(n);
  }
  // Append any remaining non-output nodes in their reported order.
  for (const n of entries) push(n);
  const nodes = ordered.concat(entries.includes("prediction") ? ["prediction"] : []);

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
            {i < nodes.length - 1 &&
              dependenciesAvailable &&
              (hasEdge(node, nodes[i + 1]) ? (
                <div className="depgraph__edge" aria-hidden>
                  <svg width="24" height="40" viewBox="0 0 24 40">
                    <line x1="12" y1="0" x2="12" y2="30" stroke="currentColor" strokeWidth="2" />
                    <path d="M12 40 L6 30 L18 30 Z" fill="currentColor" />
                  </svg>
                </div>
              ) : (
                // No declared edge between these neighbours: render a gap, not an arrow.
                <div className="depgraph__gap" aria-hidden />
              ))}
          </div>
        );
      })}
    </div>
  );
}
