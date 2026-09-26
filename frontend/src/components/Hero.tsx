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
          DriftTrace watches the loan-default feature pipeline, detects distribution
          drift per node with KS and PSI, then walks the declared dependency graph to
          name the earliest supported root cause - so an operator acts on one diagnosis
          instead of a wall of alerts.
        </p>
      </div>
      <LifecycleStrip />
    </section>
  );
}
