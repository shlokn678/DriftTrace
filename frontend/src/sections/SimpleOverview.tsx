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
  const { driftStatus, rootCause, driftedCount } = bundle;

  const hasActive = activeModel?.active ?? false;
  const modelName = hasActive ? (activeModel?.name ?? "model") : "None";
  const modelTask = activeModel?.task ?? null;
  const depsAvailable = activeModel?.dependencies_available ?? false;

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
            status={hasActive ? "stable" : "insufficient"}
            sub={
              hasActive
                ? `${modelTask ?? "model"}${depsAvailable ? " · graph provided" : " · no graph"}`
                : "Upload a model bundle to begin"
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
            ) : hasActive && report && affected.length > 0 ? (
              <Pill variant="drift" dot>
                Drift detected
              </Pill>
            ) : hasActive && report ? (
              <Pill variant="stable" dot>
                Stable
              </Pill>
            ) : (
              <Pill variant="insufficient" dot>
                No data yet
              </Pill>
            )}
          </div>

          {!hasActive ? (
            <p className="muted" style={{ maxWidth: 640 }}>
              No model is active. Upload a model bundle below, then activate it to begin
              monitoring.
            </p>
          ) : !report ? (
            <p className="muted" style={{ maxWidth: 640 }}>
              No monitoring window has run yet. Send predictions to the active model, or run a
              drift test (in View Details) to populate a real drift + root-cause result.
            </p>
          ) : affected.length === 0 ? (
            <p className="muted" style={{ maxWidth: 720 }}>
              No feature has drifted from the reference distribution. Monitoring is active.
            </p>
          ) : rootCause ? (
            <div className="stack stack-4">
              <p className="muted" style={{ maxWidth: 720 }}>
                DriftTrace traced the drift upstream and flagged{" "}
                <strong style={{ color: "var(--navy)" }}>{rootCause}</strong> as the likely
                origin. Downstream nodes are recorded as symptoms - only the root cause is
                alerted on.
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
            <div className="stack stack-4">
              <p className="muted" style={{ maxWidth: 720 }}>
                Drift detected in {affected.length} feature
                {affected.length === 1 ? "" : "s"}. Likely origin is{" "}
                <strong style={{ color: "var(--navy)" }}>undetermined</strong>
                {depsAvailable
                  ? "."
                  : " because no dependency graph was provided, so root-cause tracing is unavailable."}
              </p>
              <div className="stack stack-2">
                <MonoLabel>Affected features</MonoLabel>
                <div className="row row-3 wrap">
                  {affected.map((n) => (
                    <span key={n} className="pill pill--drift">
                      {n}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </BentoCard>
      </section>
    </div>
  );
}
