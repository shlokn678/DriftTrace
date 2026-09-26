import { Database, Boxes, LineChart, Workflow, Rocket, Radar } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import "./LifecycleStrip.css";

interface Stage {
  key: string;
  label: string;
  Icon: LucideIcon;
}

const STAGES: Stage[] = [
  { key: "data", label: "Data", Icon: Database },
  { key: "build", label: "Build", Icon: Boxes },
  { key: "track", label: "Track", Icon: LineChart },
  { key: "automate", label: "Automate", Icon: Workflow },
  { key: "deploy", label: "Deploy", Icon: Rocket },
  { key: "monitor", label: "Monitor", Icon: Radar },
];

/** The DriftTrace lifecycle markers - product markers, not a generic stepper.
 * MONITOR is the active stage (this dashboard operates the Monitor phase). */
export function LifecycleStrip() {
  return (
    <ol className="lifecycle" aria-label="ML lifecycle">
      {STAGES.map((s, i) => {
        const active = s.key === "monitor";
        return (
          <li
            key={s.key}
            className={`lifecycle__stage ${active ? "is-active" : ""}`}
            aria-current={active ? "step" : undefined}
          >
            <span className="lifecycle__marker">
              <s.Icon size={15} />
            </span>
            <span className="lifecycle__label">{s.label}</span>
            {i < STAGES.length - 1 && <span className="lifecycle__link" aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}
