import { useCallback, useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { api } from "./api/client";
import { useAsync } from "./hooks/useAsync";
import { useTheme } from "./lib/theme";
import { deriveBundle } from "./lib/derive";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { Button } from "./components/primitives";
import { Onboarding } from "./sections/Onboarding";
import { SimpleOverview } from "./sections/SimpleOverview";
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

  const report = rca.data?.available ? (rca.data.report ?? null) : null;
  const bundle = useMemo(
    () => deriveBundle(report, metrics.data, modelInfo.data),
    [report, metrics.data, modelInfo.data],
  );

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
              report={report}
              dependenciesAvailable={activeModel.data?.dependencies_available ?? false}
            />

            <section className="section bento">
              <IncidentCard report={report} />
              <ExplanationPanel features={activeModel.data?.features ?? []} />
            </section>

            <DriftMetrics report={report} />

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
