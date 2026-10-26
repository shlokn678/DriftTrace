import { useCallback, useState } from "react";
import { Activity, Play, Square } from "lucide-react";
import { api, ApiError } from "../api/client";
import { useAsync } from "../hooks/useAsync";
import type {
  LiveIncident,
  LiveStatus,
  LiveWindowResult,
} from "../api/types";
import {
  BentoCard,
  Button,
  EmptyState,
  ErrorState,
  MonoLabel,
  Pill,
} from "../components/primitives";
import { formatInt, formatNumber, type StatusKind } from "../lib/status";
import "./live.css";

interface Props {
  /** Whether a model is active (live monitoring needs one). */
  hasActiveModel: boolean;
}

const LIVE_POLL_MS = 2000;

function liveStatusVariant(status: LiveStatus): StatusKind {
  if (status === "RUNNING") return "stable";
  if (status === "ERROR") return "drift";
  return "insufficient";
}

function windowStatusVariant(status: LiveWindowResult["status"]): StatusKind {
  if (status === "DRIFT") return "drift";
  if (status === "WARNING") return "warning";
  return "stable";
}

/** Live continuous-monitoring dashboard. Polls the backend while a session runs. */
export function LiveMonitoring({ hasActiveModel }: Props) {
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const status = useAsync(() => api.liveStatus(), [], { pollMs: LIVE_POLL_MS });
  const windows = useAsync(() => api.liveWindows(30), [], { pollMs: LIVE_POLL_MS });
  const incidents = useAsync(() => api.liveIncidents(), [], { pollMs: LIVE_POLL_MS });

  const refreshAll = useCallback(() => {
    status.reload();
    windows.reload();
    incidents.reload();
  }, [status, windows, incidents]);

  const start = async () => {
    setBusy(true);
    setActionError(null);
    try {
      await api.liveStart();
      refreshAll();
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setActionError(null);
    try {
      await api.liveStop();
      refreshAll();
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const s = status.data;
  const liveStatus: LiveStatus = s?.status ?? "STOPPED";
  const running = liveStatus === "RUNNING";
  const history = windows.data?.windows ?? [];
  const latest = s?.latest_window ?? (history.length ? history[history.length - 1] : null);
  const openIncident = incidents.data?.open ?? s?.open_incident ?? null;
  const closedIncidents = incidents.data?.history ?? [];

  const windowProgress =
    s?.window_size && s.window_size > 0
      ? `${s.current_window_count ?? 0} / ${s.window_size}`
      : "-";

  return (
    <section className="section bento" id="live" aria-labelledby="live-title">
      <BentoCard className="col-12" major>
        <div className="card__head">
          <MonoLabel>
            <Activity size={14} aria-hidden style={{ verticalAlign: "-2px", marginRight: 6 }} />
            Live Monitoring
          </MonoLabel>
          <div className="row row-3">
            <Pill variant={liveStatusVariant(liveStatus)} dot>
              {liveStatus}
            </Pill>
            {!running ? (
              <Button
                variant="primary"
                onClick={start}
                disabled={busy || !hasActiveModel}
                ariaLabel="Start live monitoring"
              >
                <Play size={15} aria-hidden /> Start
              </Button>
            ) : (
              <Button variant="danger" onClick={stop} disabled={busy} ariaLabel="Stop live monitoring">
                <Square size={15} aria-hidden /> Stop
              </Button>
            )}
          </div>
        </div>

        <h2 id="live-title" style={{ marginBottom: "var(--space-4)" }}>
          Monitor the deployed model continuously
        </h2>

        {!hasActiveModel ? (
          <EmptyState
            title="No active model"
            hint="Activate a model, then start live monitoring to evaluate drift over time against its reference baseline."
          />
        ) : actionError ? (
          <ErrorState message={actionError} />
        ) : s?.error ? (
          <ErrorState message={s.error} />
        ) : (
          <div className="stack stack-5">
            {/* Counters */}
            <div className="bento live-counters">
              <Counter label="Model" value={s?.model_name ?? s?.model_id ?? "-"} />
              <Counter label="Source" value={s?.source ?? "-"} />
              <Counter label="Events received" value={formatInt(s?.events_received)} />
              <Counter label="Current window" value={windowProgress} />
              <Counter label="Windows evaluated" value={formatInt(s?.windows_evaluated)} />
              <Counter
                label="Current status"
                value={
                  latest ? (
                    <Pill variant={windowStatusVariant(latest.status)} dot>
                      {latest.status}
                    </Pill>
                  ) : (
                    "-"
                  )
                }
              />
            </div>

            {/* Latest evaluation detail */}
            {latest && <LatestWindow latest={latest} />}

            {/* Incident */}
            <IncidentPanel open={openIncident} closed={closedIncidents} />

            {/* Trend over time */}
            <TrendPanel windows={history} />

            {/* Recent window history */}
            <WindowHistory windows={history} />
          </div>
        )}
      </BentoCard>
    </section>
  );
}

function Counter({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="col-2 stack stack-2 live-counter">
      <MonoLabel>{label}</MonoLabel>
      <span className="mono-value" style={{ fontWeight: 700, color: "var(--navy)" }}>
        {value}
      </span>
    </div>
  );
}

function LatestWindow({ latest }: { latest: LiveWindowResult }) {
  return (
    <div className="stack stack-3">
      <MonoLabel>Latest evaluation ({latest.window_id})</MonoLabel>
      <div className="bento" style={{ gap: "var(--space-4)" }}>
        <Field label="Evaluated at" value={new Date(latest.evaluated_at).toLocaleTimeString()} col="col-3" />
        <Field label="Events" value={formatInt(latest.event_count)} col="col-3" />
        <Field label="Drifted features" value={formatInt(latest.drifted_count)} col="col-3" />
        <Field
          label="RCA"
          value={
            latest.dependencies_available
              ? latest.primary_root_cause ?? "(no single root)"
              : "unavailable (no graph)"
          }
          col="col-3"
        />
      </div>

      {latest.drifted_features.length > 0 && (
        <div className="stack stack-2">
          <MonoLabel>Latest drifted features</MonoLabel>
          <div className="row row-3 wrap">
            {latest.drifted_features.map((f) => (
              <span key={f} className="mono-value live-chip live-chip--drift">
                {f}
                {latest.per_feature[f]?.psi != null && (
                  <span className="tertiary"> · PSI {formatNumber(latest.per_feature[f].psi)}</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {latest.dependencies_available && latest.primary_root_cause && (
        <div className="stack stack-2">
          <MonoLabel>Root cause → symptoms</MonoLabel>
          <div className="row row-3 wrap">
            <span className="mono-value live-chip live-chip--root">{latest.primary_root_cause}</span>
            {latest.primary_symptom_path.map((sym) => (
              <span key={sym} className="row row-3">
                <span className="tertiary">→</span>
                <span className="mono-value live-chip live-chip--symptom">{sym}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function IncidentPanel({ open, closed }: { open: LiveIncident | null; closed: LiveIncident[] }) {
  return (
    <div className="stack stack-3">
      <div className="row between">
        <MonoLabel>Incident</MonoLabel>
        <Pill variant={open ? "drift" : "stable"} dot>
          {open ? "OPEN" : closed.length ? "CLOSED" : "NONE"}
        </Pill>
      </div>
      {open ? (
        <p className="muted">
          Incident <strong>{open.incident_id}</strong> open since window{" "}
          <strong>{open.opened_window}</strong>. Root cause:{" "}
          <strong>{open.root_cause ?? "undetermined"}</strong>
          {open.symptom_path.length > 0 && <> → symptoms {open.symptom_path.join(", ")}</>}. Spanning{" "}
          {open.drifted_windows} drifted window(s).
        </p>
      ) : closed.length ? (
        <p className="tertiary">
          Last incident {closed[closed.length - 1].incident_id} closed at window{" "}
          {closed[closed.length - 1].closed_window}. No active incident.
        </p>
      ) : (
        <p className="tertiary">No incident. Drift must persist across consecutive windows to open one.</p>
      )}
    </div>
  );
}

/** Simple SVG sparkline of PSI (max drifted-feature PSI) + drifted-count per window. */
function TrendPanel({ windows }: { windows: LiveWindowResult[] }) {
  if (windows.length < 2) {
    return (
      <div className="stack stack-2">
        <MonoLabel>Drift over time</MonoLabel>
        <p className="tertiary">Trend appears once at least two windows are evaluated.</p>
      </div>
    );
  }
  const psiSeries = windows.map((w) => {
    const psis = Object.values(w.per_feature)
      .map((f) => f.psi ?? 0)
      .filter((v) => Number.isFinite(v));
    return psis.length ? Math.max(...psis) : 0;
  });
  const driftSeries = windows.map((w) => w.drifted_count);
  const maxPsi = Math.max(...psiSeries, 0.3);
  const maxDrift = Math.max(...driftSeries, 1);
  const W = 100;
  const H = 32;
  const toPath = (series: number[], max: number) =>
    series
      .map((v, i) => {
        const x = (i / (series.length - 1)) * W;
        const y = H - (v / max) * H;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  return (
    <div className="stack stack-3">
      <MonoLabel>Drift over time</MonoLabel>
      <div className="bento" style={{ gap: "var(--space-4)" }}>
        <div className="col-6 stack stack-2">
          <span className="tertiary">Max PSI per window (drift threshold 0.2)</span>
          <svg className="live-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="PSI over time">
            <line
              x1="0"
              x2={W}
              y1={H - (0.2 / maxPsi) * H}
              y2={H - (0.2 / maxPsi) * H}
              className="live-spark__threshold"
            />
            <path d={toPath(psiSeries, maxPsi)} className="live-spark__line live-spark__line--psi" />
          </svg>
        </div>
        <div className="col-6 stack stack-2">
          <span className="tertiary">Drifted features per window</span>
          <svg className="live-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Drifted features over time">
            <path d={toPath(driftSeries, maxDrift)} className="live-spark__line live-spark__line--drift" />
          </svg>
        </div>
      </div>
    </div>
  );
}

function WindowHistory({ windows }: { windows: LiveWindowResult[] }) {
  if (!windows.length) {
    return (
      <div className="stack stack-2">
        <MonoLabel>Recent window history</MonoLabel>
        <p className="tertiary">No windows evaluated yet. Send prediction events to fill a window.</p>
      </div>
    );
  }
  const recent = windows.slice(-12).reverse();
  return (
    <div className="stack stack-3">
      <MonoLabel>Recent window history</MonoLabel>
      <div className="live-history">
        {recent.map((w) => (
          <div key={w.window_id} className="live-history__row">
            <span className="mono-value live-history__id">{w.window_id}</span>
            <Pill variant={windowStatusVariant(w.status)} dot>
              {w.status}
            </Pill>
            <span className="tertiary live-history__meta">
              {w.drifted_count} drifted
              {w.dependencies_available && w.primary_root_cause
                ? ` · root ${w.primary_root_cause}`
                : ""}
              {w.incident_id ? ` · ${w.incident_id}` : ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Field({ label, value, col }: { label: string; value: React.ReactNode; col: string }) {
  return (
    <div className={col}>
      <MonoLabel>{label}</MonoLabel>
      <div className="mono-value" style={{ marginTop: 6, fontWeight: 600, color: "var(--navy)" }}>
        {value}
      </div>
    </div>
  );
}
