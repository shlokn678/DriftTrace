import { useState } from "react";
import { Play, Zap, Eye, Sparkles, RotateCcw, ShieldCheck } from "lucide-react";
import { api, ApiError } from "../api/client";
import type { RunScenarioResponse, ScenarioName } from "../api/types";
import { BentoCard, Button, MonoLabel, Pill } from "../components/primitives";

interface Props {
  onScenarioComplete: () => void;
  onViewRca: () => void;
  onExplain: () => void;
}

type ActionState = { kind: "idle" | "running" | "done" | "error" | "pending"; message?: string };

/** Operator controls + deterministic demo drivers. Real backend where a workflow exists;
 * operator-approved actions honestly reflect the backend's approval semantics. */
export function OperatorActions({ onScenarioComplete, onViewRca, onExplain }: Props) {
  const [state, setState] = useState<ActionState>({ kind: "idle" });
  const [pendingAction, setPendingAction] = useState<"retrain" | "rollback" | null>(null);

  async function runScenario(scenario: ScenarioName, label: string) {
    setState({ kind: "running", message: `Running ${label}...` });
    try {
      const res: RunScenarioResponse = await api.runScenario(scenario);
      const out = res.outcome;
      const summary = out?.has_root_cause
        ? `Root cause: ${out.root_cause_candidates.join(", ")}`
        : "No drift detected";
      setState({ kind: "done", message: `${label} complete. ${summary}.` });
      onScenarioComplete();
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof ApiError ? err.message : (err as Error).message,
      });
    }
  }

  return (
    <BentoCard className="col-12" major id="operator">
      <div className="card__head">
        <div>
          <MonoLabel>Operator</MonoLabel>
          <h2 style={{ marginTop: 6 }}>Run the demo. Decide the response.</h2>
        </div>
        {state.kind !== "idle" && (
          <Pill
            variant={state.kind === "error" ? "drift" : state.kind === "done" ? "stable" : "accent"}
            dot
          >
            {state.kind === "running"
              ? "Working"
              : state.kind === "done"
                ? "Complete"
                : state.kind === "pending"
                  ? "Awaiting approval"
                  : "Error"}
          </Pill>
        )}
      </div>

      <div className="op-grid">
        <div className="op-group">
          <MonoLabel>Deterministic demo</MonoLabel>
          <div className="row row-3 wrap" style={{ marginTop: "var(--space-3)" }}>
            <Button variant="default" onClick={() => runScenario("control", "Normal run")} disabled={state.kind === "running"}>
              <Play size={16} /> Run Normal
            </Button>
            <Button variant="primary" onClick={() => runScenario("income_annual", "Income drift")} disabled={state.kind === "running"}>
              <Zap size={16} /> Simulate Income Drift
            </Button>
            <Button variant="default" onClick={() => runScenario("mid_chain", "Mid-chain drift")} disabled={state.kind === "running"}>
              Mid-chain
            </Button>
            <Button variant="default" onClick={() => runScenario("two_roots", "Two-root")} disabled={state.kind === "running"}>
              Two roots
            </Button>
          </div>
          <p className="tertiary" style={{ fontSize: "0.8rem", marginTop: "var(--space-3)" }}>
            Runs the real KS/PSI + graph RCA pipeline on the backend and writes the latest
            monitoring report. Not a frontend-only simulation.
          </p>
        </div>

        <div className="op-group">
          <MonoLabel>Inspect</MonoLabel>
          <div className="row row-3 wrap" style={{ marginTop: "var(--space-3)" }}>
            <Button variant="default" onClick={onViewRca}>
              <Eye size={16} /> View RCA
            </Button>
            <Button variant="default" onClick={onExplain}>
              <Sparkles size={16} /> Explain Prediction
            </Button>
          </div>
        </div>

        <div className="op-group">
          <MonoLabel>High-impact (approval required)</MonoLabel>
          <div className="row row-3 wrap" style={{ marginTop: "var(--space-3)" }}>
            <Button
              variant="default"
              onClick={() => {
                setPendingAction("retrain");
                setState({ kind: "pending" });
              }}
            >
              <ShieldCheck size={16} /> Approve Retrain
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                setPendingAction("rollback");
                setState({ kind: "pending" });
              }}
            >
              <RotateCcw size={16} /> Rollback
            </Button>
          </div>
        </div>
      </div>

      {pendingAction && (
        <div className="op-confirm" role="alertdialog" aria-label={`Confirm ${pendingAction}`}>
          <div className="stack stack-2">
            <strong style={{ color: "var(--navy)" }}>
              Confirm operator {pendingAction}
            </strong>
            <p className="muted" style={{ fontSize: "0.88rem" }}>
              DriftTrace never rolls back or retrains automatically. This action requires
              explicit operator approval and is recorded in the audit trail. Execute it via
              the approved backend command:
            </p>
            <code className="op-cmd">
              {pendingAction === "retrain"
                ? "drifttrace retrain --approve --approver <you>"
                : "drifttrace rollback --to-version <v> --approve --approver <you>"}
            </code>
          </div>
          <div className="row row-3" style={{ marginTop: "var(--space-3)" }}>
            <Button
              variant={pendingAction === "rollback" ? "danger" : "primary"}
              onClick={() => {
                setState({
                  kind: "done",
                  message: `${pendingAction === "retrain" ? "Retrain" : "Rollback"} acknowledged - run the approved command to execute (audited).`,
                });
                setPendingAction(null);
              }}
            >
              I understand
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setPendingAction(null);
                setState({ kind: "idle" });
              }}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      {state.message && !pendingAction && (
        <p
          className="op-status"
          style={{ color: state.kind === "error" ? "var(--status-drift)" : "var(--text-secondary)" }}
        >
          {state.message}
        </p>
      )}
    </BentoCard>
  );
}
