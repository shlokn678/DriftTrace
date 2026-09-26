import type { HealthResponse, Metrics, ModelInfoResponse, ReadyResponse } from "../api/types";
import { BentoCard, MonoLabel, Pill } from "../components/primitives";
import { formatInt } from "../lib/status";

interface Props {
  health: HealthResponse | null;
  ready: ReadyResponse | null;
  modelInfo: ModelInfoResponse | null;
  metrics: Metrics | null;
}

/** Compact operations area: API health, readiness, event processing, metrics. */
export function SystemHealth({ health, ready, modelInfo, metrics }: Props) {
  const rows: { label: string; value: string; ok?: boolean }[] = [
    { label: "API health", value: health?.status ?? "unavailable", ok: health?.status === "ok" },
    { label: "Readiness", value: ready?.ready ? "ready" : "not ready", ok: ready?.ready },
    { label: "Model loaded", value: modelInfo?.loaded ? "loaded" : "not loaded", ok: modelInfo?.loaded },
    { label: "Model name", value: modelInfo?.model_name ?? "-" },
  ];

  const counters: { label: string; key: string }[] = [
    { label: "Predictions", key: "drifttrace_prediction_requests_total" },
    { label: "Events", key: "drifttrace_prediction_events_total" },
    { label: "Windows", key: "drifttrace_windows_processed_total" },
    { label: "Drift checks", key: "drifttrace_drift_checks_total" },
    { label: "Drifted nodes", key: "drifttrace_drifted_nodes_total" },
    { label: "Alerts", key: "drifttrace_root_cause_alerts_total" },
  ];

  return (
    <BentoCard className="col-12" id="system">
      <div className="card__head">
        <MonoLabel>System / Metrics</MonoLabel>
        <Pill variant="accent">/metrics</Pill>
      </div>
      <div className="bento">
        <div className="col-4 stack stack-3">
          {rows.map((r) => (
            <div key={r.label} className="row between sys-row">
              <MonoLabel>{r.label}</MonoLabel>
              <Pill variant={r.ok === undefined ? undefined : r.ok ? "stable" : "drift"} dot={r.ok !== undefined}>
                {r.value}
              </Pill>
            </div>
          ))}
        </div>
        <div className="col-8">
          <div className="bento" style={{ gap: "var(--space-4)" }}>
            {counters.map((c) => (
              <div className="col-4 sys-counter" key={c.key}>
                <MonoLabel>{c.label}</MonoLabel>
                <div className="mono-value" style={{ fontSize: "1.4rem", fontWeight: 700, color: "var(--navy)", marginTop: 4 }}>
                  {metrics ? formatInt(metrics[c.key] ?? 0) : "-"}
                </div>
              </div>
            ))}
          </div>
          <p className="tertiary" style={{ fontSize: "0.78rem", marginTop: "var(--space-4)" }}>
            Prometheus-compatible counters (Prometheus/Grafana deferred as stretch). API-process
            counters; the monitor process maintains its own.
          </p>
        </div>
      </div>
    </BentoCard>
  );
}
