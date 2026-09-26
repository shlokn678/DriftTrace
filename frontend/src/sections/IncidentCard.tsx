import type { MonitoringReport } from "../api/types";
import { BentoCard, EmptyState, MonoLabel, Pill } from "../components/primitives";

interface Props {
  report: MonitoringReport | null;
}

/** The latest incident / root-cause alert, straight from the real backend report. */
export function IncidentCard({ report }: Props) {
  const alert = report?.alerts?.[0] ?? null;

  return (
    <BentoCard className="col-5" major>
      <div className="card__head">
        <MonoLabel>Root-Cause Alert</MonoLabel>
        {alert ? (
          <Pill variant="root" dot>
            Incident open
          </Pill>
        ) : (
          <Pill variant="stable" dot>
            No incident
          </Pill>
        )}
      </div>

      {!alert ? (
        <EmptyState
          title="No active incident"
          hint="Root-cause-only alerting emits at most one alert per causal chain. Symptoms are recorded as evidence, not alerts."
        />
      ) : (
        <div className="stack stack-5">
          <div>
            <MonoLabel>Root cause</MonoLabel>
            <div
              className="mono-value"
              style={{ fontSize: "2rem", fontWeight: 800, color: "var(--status-root)", marginTop: 4 }}
            >
              {alert.root_cause}
            </div>
          </div>

          <div className="stack stack-2">
            <MonoLabel>Downstream symptoms</MonoLabel>
            <div className="row row-3 wrap">
              {alert.symptom_path.length ? (
                alert.symptom_path.map((s) => (
                  <span
                    key={s}
                    className="mono-value"
                    style={{
                      padding: "5px 12px",
                      borderRadius: "var(--radius-pill)",
                      background: "var(--status-drift-surface)",
                      color: "var(--status-drift)",
                      fontWeight: 600,
                      fontSize: "0.86rem",
                    }}
                  >
                    {s}
                  </span>
                ))
              ) : (
                <span className="tertiary">none on this chain</span>
              )}
            </div>
          </div>

          <hr className="divider" />
          <div className="bento" style={{ gap: "var(--space-4)" }}>
            <Field label="Incident" value={alert.incident_id} col="col-6" />
            <Field label="Window" value={alert.window_id} col="col-6" />
            <Field label="Model" value={`v${alert.model_version}`} col="col-6" />
            <Field label="Delivery" value={alert.delivery_status} col="col-6" />
          </div>
        </div>
      )}
    </BentoCard>
  );
}

function Field({ label, value, col }: { label: string; value: string; col: string }) {
  return (
    <div className={col}>
      <MonoLabel>{label}</MonoLabel>
      <div
        className="mono-value"
        style={{ marginTop: 4, fontWeight: 600, color: "var(--navy)", wordBreak: "break-all" }}
      >
        {value}
      </div>
    </div>
  );
}
