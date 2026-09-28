import type { ActiveModelResponse, MonitoringReport } from "../api/types";
import type { MetricsBundle } from "../lib/derive";
import { BentoCard, MetricCard, MonoLabel, Pill } from "../components/primitives";

interface Props {
  bundle: MetricsBundle;
  report: MonitoringReport | null;
  healthy: boolean | null;
  ready: boolean | null;
  activeModel: ActiveModelResponse | null;
}

/**
 * The default, minimal-input operator view (Phase 5). Answers the only questions a
 * normal user needs at a glance: which model is active, is the system up, is there
 * drift, what is the root cause, and which features are affected. Everything technical
 * lives behind View Details.
 */
export function SimpleOverview({ bundle, report, healthy, ready, activeModel }: Props) {
  const { modelVersion, driftStatus, rootCause, driftedCount } = bundle;

  const modelName = activeModel?.name ?? "drifttrace-loan-default";
  const isCustom = activeModel?.is_custom ?? false;

  const systemStatus =
    healthy === false
      ? { text: "Offline", status: "drift" as const }
      : healthy === null
        ? { text: "Checking", status: "insufficient" as const }
        : ready
          ? { text: "Operational", status: "stable" as const }
          : { text: "Not ready", status: "warning" as const };

  const affected = report?.drift.drifted_nodes ?? [];

  return (
    <div className="stack stack-5" id="overview">
      <section className="section bento" aria-label="System overview">
        <div className="col-3">
          <MetricCard
            label="Active Model"
            value={<span style={{ fontSize: "1.15rem", wordBreak: "break-word" }}>{modelName}</span>}
            sub={
              isCustom
                ? "Uploaded model (custom)"
                : `Default loan model${modelVersion ? ` · v${modelVersion}` : ""}`
            }
          />
        </div>
        <div className="col-3">
          <MetricCard
            label="System Status"
            value={systemStatus.text}
            status={systemStatus.status}
            sub="API + monitor"
          />
        </div>
        <div className="col-3">
          <MetricCard
            label="Drift Status"
            value={driftStatus.text}
            status={driftStatus.status}
            sub={`${driftedCount} drifted node${driftedCount === 1 ? "" : "s"}`}
          />
        </div>
        <div className="col-3">
          <MetricCard
            label="Root Cause"
            value={rootCause ?? "None"}
            status={rootCause ? "root" : "stable"}
            sub={rootCause ? "Earliest supported node" : "No candidate"}
          />
        </div>
      </section>

      <section className="section bento" aria-label="Diagnosis">
        <BentoCard className="col-12" major>
          <div className="card__head">
            <MonoLabel>Diagnosis</MonoLabel>
            {rootCause ? (
              <Pill variant="root" dot>
                Root cause identified
              </Pill>
            ) : report ? (
              <Pill variant="stable" dot>
                No root cause
              </Pill>
            ) : (
              <Pill variant="insufficient" dot>
                No data yet
              </Pill>
            )}
          </div>

          {!report ? (
            <p className="muted" style={{ maxWidth: 640 }}>
              No monitoring window has run yet. Run a scenario from the operator controls (in
              Advanced) to populate a real drift + root-cause result.
            </p>
          ) : rootCause ? (
            <div className="stack stack-4">
              <p className="muted" style={{ maxWidth: 720 }}>
                DriftTrace traced the drift alarms upstream and flagged{" "}
                <strong style={{ color: "var(--navy)" }}>{rootCause}</strong> as the earliest
                drifted node. Downstream nodes are recorded as symptoms - only the root cause
                is alerted on.
              </p>
              <div className="stack stack-2">
                <MonoLabel>Affected features</MonoLabel>
                <div className="row row-3 wrap">
                  {affected.map((n) => (
                    <span
                      key={n}
                      className={`pill ${n === rootCause ? "pill--root" : "pill--drift"}`}
                    >
                      {n === rootCause ? `${n} (root cause)` : n}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="muted" style={{ maxWidth: 720 }}>
              The latest window shows no drifted node without a drifted ancestor. No root-cause
              alert was raised.
            </p>
          )}
        </BentoCard>
      </section>
    </div>
  );
}
