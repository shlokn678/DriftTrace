import { MonoLabel } from "./primitives";
import { LifecycleStrip } from "./LifecycleStrip";
import "./Hero.css";

export function Hero() {
  return (
    <section className="hero" id="overview" aria-labelledby="hero-title">
      <div className="hero__text">
        <MonoLabel>Model Operations / Root-Cause Analysis</MonoLabel>
        <h1 id="hero-title">
          Detect drift.
          <br />
          Trace the cause.
        </h1>
        <p className="hero__lede muted">
          Upload any supported model with its reference data. DriftTrace detects
          distribution drift per feature with KS and PSI, then - when a dependency graph
          is provided - traces upstream to the likely root cause, so an operator acts on
          one diagnosis instead of a wall of alerts.
        </p>
      </div>
      <LifecycleStrip />
    </section>
  );
}
