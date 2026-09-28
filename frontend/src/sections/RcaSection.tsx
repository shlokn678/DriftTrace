import type { MonitoringReport } from "../api/types";
import { BentoCard, EmptyState, MonoLabel, Pill } from "../components/primitives";
import { DependencyGraph } from "../components/DependencyGraph";
import { formatNumber } from "../lib/status";

interface Props {
  report: MonitoringReport | null;
  dependenciesAvailable?: boolean;
}

/** The primary RCA section: dependency graph + root-cause reasoning/evidence. */
export function RcaSection({ report, dependenciesAvailable = true }: Props) {
  const rca = report?.rca ?? null;
  const roots = rca?.root_cause_candidates ?? [];
  const primary = roots[0];
  const drifted = report?.drift.drifted_nodes ?? [];

  return (
    <section className="section bento" id="rca" aria-labelledby="rca-title">
      <BentoCard className="col-5" major>
        <div className="card__head">
          <MonoLabel>Dependency Graph</MonoLabel>
          <Pill variant="accent">NetworkX RCA</Pill>
        </div>
        <h2 id="rca-title" style={{ marginBottom: "var(--space-5)" }}>
          Trace upstream
        </h2>
        <DependencyGraph
          classifications={report?.classifications ?? null}
          dependenciesAvailable={dependenciesAvailable}
        />
      </BentoCard>

      <BentoCard className="col-7" major>
        <div className="card__head">
          <MonoLabel>Root-Cause Reasoning</MonoLabel>
          {rca?.has_root_cause ? (
            <Pill variant="root" dot>
              Root cause identified
            </Pill>
          ) : (
            <Pill variant="stable" dot>
              No root cause
            </Pill>
          )}
        </div>

        {!report ? (
          <EmptyState
            title="No monitoring window yet"
            hint="Send predictions or run a drift test to populate the latest RCA from the real KS/PSI + graph analysis."
          />
        ) : !rca?.has_root_cause ? (
          <div className="stack stack-4">
            <p className="muted">
              {!dependenciesAvailable && drifted.length > 0
                ? "Drift was detected, but no dependency graph was provided for this model, so dependency-based root-cause tracing is unavailable. The affected features are listed below."
                : drifted.length > 0
                  ? "The latest window shows no drifted feature without a drifted ancestor, so no single root cause was isolated."
                  : "The latest monitoring window shows no drift. No root-cause alert was emitted."}
            </p>
            {drifted.length > 0 && (
              <div className="stack stack-2">
                <MonoLabel>Affected features</MonoLabel>
                <div className="row row-3 wrap">
                  {drifted.map((n) => (
                    <span key={n} className="mono-value" style={symptomChip}>
                      {n}
                    </span>
                  ))}
                </div>
              </div>
            )}
            <RcaMeta report={report} />
          </div>
        ) : (
          <div className="stack stack-4">
            <div className="stack stack-2">
              <MonoLabel>Earliest supported root cause</MonoLabel>
              <div className="row row-3 wrap">
                {roots.map((r) => (
                  <span key={r.node} className="mono-value" style={rootChip}>
                    {r.node}
                  </span>
                ))}
              </div>
            </div>

            {primary && primary.symptom_path.length > 0 && (
              <div className="stack stack-2">
                <MonoLabel>Downstream symptom path</MonoLabel>
                <div className="row row-3 wrap">
                  {primary.symptom_path.map((s, i) => (
                    <span key={s} className="row row-3">
                      <span className="mono-value" style={symptomChip}>
                        {s}
                      </span>
                      {i < primary.symptom_path.length - 1 && (
                        <span className="tertiary">-&gt;</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <RcaMeta report={report} />
          </div>
        )}
      </BentoCard>
    </section>
  );
}

function RcaMeta({ report }: { report: MonitoringReport }) {
  // Evidence for the primary root cause, else the first drifted node - fully dynamic.
  const primaryNode =
    report.rca.root_cause_candidates?.[0]?.node ?? report.drift.drifted_nodes?.[0] ?? null;
  const node = primaryNode ? report.drift.nodes[primaryNode] : undefined;
  const ks = node?.ks;
  const psi = node?.psi;
  return (
    <>
      <hr className="divider" />
      <div className="bento" style={{ gap: "var(--space-4)" }}>
        <Meta label="Monitoring window" value={report.window_id} col="col-3" />
        <Meta label="Model" value={report.model_version} col="col-3" />
        <Meta label="Baseline" value={report.baseline_version} col="col-3" />
        <Meta label="Incident" value={report.incident_id ?? "-"} col="col-3" />
      </div>
      {primaryNode && ks && psi && (
        <p className="tertiary" style={{ fontSize: "0.82rem" }}>
          Evidence ({primaryNode}): KS p-value {formatNumber(ks.p_value)} vs threshold{" "}
          {formatNumber(ks.threshold, 2)}; PSI {formatNumber(psi.psi)} vs drift threshold{" "}
          {formatNumber(psi.psi_drift, 2)}.
        </p>
      )}
    </>
  );
}

function Meta({ label, value, col }: { label: string; value: string; col: string }) {
  return (
    <div className={col}>
      <MonoLabel>{label}</MonoLabel>
      <div className="mono-value" style={{ marginTop: 6, fontWeight: 600, color: "var(--navy)" }}>
        {value}
      </div>
    </div>
  );
}

const rootChip: React.CSSProperties = {
  padding: "6px 14px",
  borderRadius: "var(--radius-pill)",
  background: "var(--status-root)",
  color: "var(--surface)",
  fontWeight: 700,
  fontSize: "0.95rem",
};

const symptomChip: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: "var(--radius-pill)",
  background: "var(--status-drift-surface)",
  color: "var(--status-drift)",
  fontWeight: 600,
  fontSize: "0.9rem",
};
