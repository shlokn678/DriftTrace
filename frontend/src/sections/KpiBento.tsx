import type { MetricsBundle } from "../lib/derive";
import { MetricCard } from "../components/primitives";

interface Props {
  bundle: MetricsBundle;
}

/** The KPI / system-health bento row. Real values only; "-" when unavailable. */
export function KpiBento({ bundle }: Props) {
  const {
    modelVersion,
    driftStatus,
    rootCause,
    activeAlerts,
    driftedCount,
    windowsProcessed,
  } = bundle;

  return (
    <section className="section bento" id="kpi" aria-label="System health metrics">
      <div className="col-3">
        <MetricCard label="Model Version" value={modelVersion ? `v${modelVersion}` : "-"} sub="Registered model" />
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
          sub={rootCause ? "Earliest supported" : "No candidate"}
        />
      </div>
      <div className="col-3">
        <MetricCard
          label="Active Alerts"
          value={activeAlerts}
          status={activeAlerts > 0 ? "root" : "stable"}
          sub="Root-cause only"
        />
      </div>
      <div className="col-6">
        <MetricCard
          label="Monitoring Windows"
          value={windowsProcessed}
          sub="Processed by the monitor (this process)"
        />
      </div>
      <div className="col-6">
        <MetricCard
          label="Predictions Served"
          value={bundle.predictionRequests}
          sub={`${bundle.predictionEvents} prediction events emitted`}
        />
      </div>
    </section>
  );
}
