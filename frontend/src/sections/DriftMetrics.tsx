import type { MonitoringReport } from "../api/types";
import { BentoCard, EmptyState, MonoLabel, StatusPill } from "../components/primitives";
import { formatInt, formatNumber, verdictToStatus, NODE_LABELS } from "../lib/status";

interface Props {
  report: MonitoringReport | null;
}

/** Per-node KS / PSI diagnostics from the real drift report. Large numbers, mono labels. */
export function DriftMetrics({ report }: Props) {
  const nodes = report ? Object.values(report.drift.nodes) : [];

  return (
    <section className="section" id="diagnostics" aria-labelledby="diag-title">
      <div className="card__head" style={{ marginBottom: "var(--space-4)" }}>
        <div>
          <MonoLabel>Drift Diagnostics</MonoLabel>
          <h2 id="diag-title" style={{ marginTop: 6 }}>
            KS + PSI per node
          </h2>
        </div>
      </div>

      {!report ? (
        <BentoCard>
          <EmptyState title="No diagnostics yet" hint="Run a scenario to compute KS/PSI." />
        </BentoCard>
      ) : (
        <div className="bento">
          {nodes.map((n) => {
            const ks = n.ks;
            const psi = n.psi;
            const status = verdictToStatus(n.verdict);
            return (
              <BentoCard className="col-4" key={n.node} interactive>
                <div className="card__head">
                  <span className="mono-value" style={{ fontWeight: 700, color: "var(--navy)" }}>
                    {NODE_LABELS[n.node] ?? n.node}
                  </span>
                  <StatusPill status={status} />
                </div>
                <div className="bento" style={{ gap: "var(--space-4)" }}>
                  <Stat label="KS statistic" value={formatNumber(ks?.statistic)} />
                  <Stat label="KS p-value" value={formatNumber(ks?.p_value)} />
                  <Stat label="PSI" value={formatNumber(psi?.psi)} />
                  <Stat
                    label="Samples"
                    value={formatInt(ks?.current_n ?? psi?.current_n)}
                  />
                </div>
                <p className="tertiary" style={{ fontSize: "0.78rem", marginTop: "var(--space-3)" }}>
                  Thresholds: KS p &lt; {formatNumber(ks?.threshold, 2)}, PSI drift &gt;={" "}
                  {formatNumber(psi?.psi_drift, 2)}
                </p>
              </BentoCard>
            );
          })}
        </div>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="col-6">
      <MonoLabel>{label}</MonoLabel>
      <div
        className="mono-value"
        style={{ fontWeight: 700, fontSize: "1.5rem", color: "var(--navy)", marginTop: 4 }}
      >
        {value}
      </div>
    </div>
  );
}
