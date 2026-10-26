import { useCallback, useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { api } from "./api/client";
import { useAsync } from "./hooks/useAsync";
import { useTheme } from "./lib/theme";
import { deriveBundle } from "./lib/derive";
import { liveWindowToReport } from "./lib/liveAdapter";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { Button } from "./components/primitives";
import { Onboarding } from "./sections/Onboarding";
import { SimpleOverview } from "./sections/SimpleOverview";
import { LiveMonitoring } from "./sections/LiveMonitoring";
import { RcaSection } from "./sections/RcaSection";
import { DriftMetrics } from "./sections/DriftMetrics";
import { IncidentCard } from "./sections/IncidentCard";
import { ExplanationPanel } from "./sections/ExplanationPanel";
import { OperatorActions } from "./sections/OperatorActions";
import { SystemHealth } from "./sections/SystemHealth";
import "./sections/sections.css";

export default function App() {
  const [nonce, setNonce] = useState(0);
  const bump = useCallback(() => setNonce((n) => n + 1), []);
  const { theme, toggle: toggleTheme } = useTheme();
  const [advanced, setAdvanced] = useState(false);

  const health = useAsync(() => api.health(), [nonce], { pollMs: 15000 });
  const ready = useAsync(() => api.ready(), [nonce], { pollMs: 15000 });
  const modelInfo = useAsync(() => api.modelInfo(), [nonce]);
  const activeModel = useAsync(() => api.activeModel(), [nonce]);
  const rca = useAsync(() => api.rcaLatest(), [nonce]);
  const metrics = useAsync(() => api.metrics(), [nonce], { pollMs: 15000 });

  // Live polling (mirrors LiveMonitoring's 2000ms cadence) so the advanced panels can
  // reflect the live session when one is running for the active model.
  const liveStatus = useAsync(() => api.liveStatus(), [nonce], { pollMs: 2000 });
  const liveLatest = useAsync(() => api.liveLatest(), [nonce], { pollMs: 2000 });
  const liveIncidents = useAsync(() => api.liveIncidents(), [nonce], { pollMs: 2000 });

  const report = rca.data?.available ? (rca.data.report ?? null) : null;
  const bundle = useMemo(
    () => deriveBundle(report, metrics.data, modelInfo.data),
    [report, metrics.data, modelInfo.data],
  );

  // Source precedence: use the live stream only when a live session for the CURRENTLY
  // ACTIVE model has evaluated at least one window (model_id must match), otherwise fall
  // back to the batch /rca/latest report. Never mix or leak across models.
  const activeId = activeModel.data?.active ? (activeModel.data.model_id ?? null) : null;
  const { panelReport, source, liveDependenciesAvailable } = useMemo(() => {
    const liveUsable =
      liveStatus.data?.active_model === true &&
      liveStatus.data.model_id === activeId &&
      (liveStatus.data.windows_evaluated ?? 0) >= 1 &&
      liveLatest.data?.available === true &&
      !!liveLatest.data.window &&
      liveLatest.data.window.model_id === activeId;
    if (liveUsable && liveLatest.data?.window) {
      const window = liveLatest.data.window;
      const incident =
        liveIncidents.data?.open ??
        liveIncidents.data?.history?.[liveIncidents.data.history.length - 1] ??
        null;
      return {
        panelReport: liveWindowToReport(window, incident),
        source: "live" as const,
        liveDependenciesAvailable: window.dependencies_available,
      };
    }
    return {
      panelReport: report,
      source: "batch" as const,
      liveDependenciesAvailable: false,
    };
  }, [liveStatus.data, liveLatest.data, liveIncidents.data, activeId, report]);

  const dependenciesAvailable =
    source === "live"
      ? liveDependenciesAvailable
      : (activeModel.data?.dependencies_available ?? false);

  const refreshing =
    health.loading ||
    ready.loading ||
    modelInfo.loading ||
    activeModel.loading ||
    rca.loading ||
    metrics.loading;

  const scrollTo = (id: string) => () => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const healthy = health.error ? false : health.data ? true : null;

  return (
    <div className="shell">
      <Header
        healthy={healthy}
        ready={ready.data?.ready ?? null}
        modelName={activeModel.data?.active ? (activeModel.data.name ?? "model") : null}
        onRefresh={bump}
        refreshing={refreshing}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="container" style={{ paddingBottom: "var(--space-8)" }}>
        <Hero />

        {/* Simplified default view: model, system status, drift status, root cause, affected features. */}
        <SimpleOverview
          bundle={bundle}
          report={report}
          healthy={healthy}
          ready={ready.data?.ready ?? null}
          activeModel={activeModel.data}
        />

        {/* Onboard a model with minimal input, then activate it ("Use this model"). */}
        <Onboarding onReady={bump} onActivated={bump} />

        {/* Live continuous monitoring of the active model (additive; polls while running). */}
        <LiveMonitoring hasActiveModel={activeModel.data?.active ?? false} />

        {/* Progressive disclosure: everything technical lives behind Advanced. */}
        <section className="section" aria-label="Advanced details toggle">
          <Button
            variant={advanced ? "ghost" : "primary"}
            onClick={() => setAdvanced((v) => !v)}
            ariaLabel={advanced ? "Hide advanced details" : "Show advanced details"}
          >
            {advanced ? (
              <>
                <ChevronUp size={16} aria-hidden /> Hide advanced details
              </>
            ) : (
              <>
                <ChevronDown size={16} aria-hidden /> View details (KS/PSI, graph, SHAP/LIME,
                operator, health)
              </>
            )}
          </Button>
        </section>

        {advanced && (
          <div className="stack stack-5">
            <RcaSection
              report={panelReport}
              dependenciesAvailable={dependenciesAvailable}
              source={source}
            />

            <section className="section bento">
              <IncidentCard report={panelReport} source={source} />
              <ExplanationPanel features={activeModel.data?.features ?? []} />
            </section>

            <DriftMetrics report={panelReport} source={source} />

            <section className="section bento">
              <OperatorActions
                onDriftTestComplete={bump}
                onViewRca={scrollTo("rca")}
                onExplain={scrollTo("diagnostics")}
                hasActiveModel={activeModel.data?.active ?? false}
              />
            </section>

            <section className="section bento">
              <SystemHealth
                health={health.data}
                ready={ready.data}
                modelInfo={modelInfo.data}
                metrics={metrics.data}
              />
            </section>
          </div>
        )}

        <footer className="site-footer">
          <span className="mono-label">DriftTrace</span>
          <span className="tertiary">
            ML drift detection - dependency-graph root-cause analysis - operator-approved
            response. Local / internal use only.
          </span>
        </footer>
      </main>
    </div>
  );
}
