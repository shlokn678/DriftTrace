import { useCallback, useMemo, useState } from "react";
import { api } from "./api/client";
import { useAsync } from "./hooks/useAsync";
import { deriveBundle } from "./lib/derive";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { KpiBento } from "./sections/KpiBento";
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

  const health = useAsync(() => api.health(), [nonce], { pollMs: 15000 });
  const ready = useAsync(() => api.ready(), [nonce], { pollMs: 15000 });
  const modelInfo = useAsync(() => api.modelInfo(), [nonce]);
  const rca = useAsync(() => api.rcaLatest(), [nonce]);
  const metrics = useAsync(() => api.metrics(), [nonce], { pollMs: 15000 });

  const report = rca.data?.available ? (rca.data.report ?? null) : null;
  const bundle = useMemo(
    () => deriveBundle(report, metrics.data, modelInfo.data),
    [report, metrics.data, modelInfo.data],
  );

  const refreshing =
    health.loading || ready.loading || modelInfo.loading || rca.loading || metrics.loading;

  const scrollTo = (id: string) => () => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="shell">
      <Header
        healthy={health.error ? false : health.data ? true : null}
        ready={ready.data?.ready ?? null}
        modelVersion={bundle.modelVersion}
        onRefresh={bump}
        refreshing={refreshing}
      />

      <main className="container" style={{ paddingBottom: "var(--space-8)" }}>
        <Hero />

        <KpiBento bundle={bundle} />

        <RcaSection report={report} />

        <section className="section bento">
          <IncidentCard report={report} />
          <ExplanationPanel />
        </section>

        <DriftMetrics report={report} />

        <section className="section bento">
          <OperatorActions
            onScenarioComplete={bump}
            onViewRca={scrollTo("rca")}
            onExplain={scrollTo("diagnostics")}
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
