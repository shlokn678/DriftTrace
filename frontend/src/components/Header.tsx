import { Activity, RefreshCw } from "lucide-react";
import { MonoLabel, Pill } from "./primitives";
import "./Header.css";

interface HeaderProps {
  healthy: boolean | null;
  ready: boolean | null;
  modelVersion: string | null;
  onRefresh: () => void;
  refreshing: boolean;
}

export function Header({
  healthy,
  ready,
  modelVersion,
  onRefresh,
  refreshing,
}: HeaderProps) {
  const statusVariant = healthy === false ? "drift" : healthy === null ? "insufficient" : "stable";
  const statusText =
    healthy === false ? "API down" : healthy === null ? "Checking" : ready ? "Operational" : "Not ready";

  return (
    <header className="header">
      <div className="container header__inner">
        <div className="header__brand">
          <div className="header__mark" aria-hidden>
            <Activity size={18} strokeWidth={2.4} />
          </div>
          <div className="stack">
            <span className="header__word">DriftTrace</span>
            <MonoLabel>ML Operations / Root-Cause Analysis</MonoLabel>
          </div>
        </div>

        <nav className="header__nav" aria-label="Primary">
          <a href="#overview" className="header__link">Overview</a>
          <a href="#rca" className="header__link">Root Cause</a>
          <a href="#diagnostics" className="header__link">Diagnostics</a>
          <a href="#operator" className="header__link">Operator</a>
        </nav>

        <div className="header__meta">
          <Pill variant={statusVariant} dot>
            {statusText}
          </Pill>
          <Pill variant="accent">MODEL v{modelVersion ?? "-"}</Pill>
          <button
            className="header__refresh"
            onClick={onRefresh}
            disabled={refreshing}
            aria-label="Refresh dashboard data"
            title="Refresh"
          >
            <RefreshCw size={16} className={refreshing ? "spin" : ""} aria-hidden />
          </button>
        </div>
      </div>
    </header>
  );
}
