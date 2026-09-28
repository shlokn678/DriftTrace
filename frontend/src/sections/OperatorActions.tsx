import { useState } from "react";
import { Zap, Eye, Sparkles, RotateCcw, ShieldCheck } from "lucide-react";
import { api, ApiError } from "../api/client";
import type { DriftTestResponse } from "../api/types";
import { BentoCard, Button, MonoLabel, Pill } from "../components/primitives";

interface Props {
  onDriftTestComplete: () => void;
  onViewRca: () => void;
  onExplain: () => void;
  /** Whether a model is active (drift test requires one). */
  hasActiveModel: boolean;
}

type ActionState = { kind: "idle" | "running" | "done" | "error" | "pending"; message?: string };

/** Operator controls: a generic drift test against the active model's reference data,
 * plus operator-approved actions that honestly reflect the backend's approval semantics.
 * The frontend never computes drift/RCA - it only triggers the real backend pipeline. */
export function OperatorActions({
  onDriftTestComplete,
  onViewRca,
  onExplain,
  hasActiveModel,
}: Props) {
  const [state, setState] = useState<ActionState>({ kind: "idle" });
  const [pendingAction, setPendingAction] = useState<"retrain" | "rollback" | null>(null);

  async function runDriftTest(intensity: number, label: string) {
    setState({ kind: "running", message: `Running ${label}...` });
    try {
      const res: DriftTestResponse = await api.runDriftTest(intensity);
      const out = res.outcome;
      const summary = !out
        ? "No result"
        : out.has_root_cause
          ? `Likely origin: ${out.root_cause_candidates.join(", ")}`
          : out.drifted_nodes.length > 0
            ? `Drift in ${out.drifted_nodes.length} feature(s); origin undetermined${
                res.dependencies_available ? "" : " (no graph)"
              }`
            : "No drift detected";
      setState({ kind: "done", message: `${label} complete. ${summary}.` });
      onDriftTestComplete();
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
          <h2 style={{ marginTop: 6 }}>Test for drift. Decide the response.</h2>
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
          <MonoLabel>Drift test</MonoLabel>
          <div className="row row-3 wrap" style={{ marginTop: "var(--space-3)" }}>
            <Button
              variant="primary"
              onClick={() => runDriftTest(2, "Drift test")}
              disabled={state.kind === "running" || !hasActiveModel}
            >
              <Zap size={16} /> Run Drift Test
            </Button>
            <Button
              variant="default"
              onClick={() => runDriftTest(0, "Baseline test")}
              disabled={state.kind === "running" || !hasActiveModel}
            >
              Run Baseline (no shift)
            </Button>
          </div>
          <p className="tertiary" style={{ fontSize: "0.8rem", marginTop: "var(--space-3)" }}>
            {hasActiveModel
              ? "Perturbs a sample of the active model's reference data and runs it through the real KS/PSI + graph RCA pipeline. Not a frontend-only simulation."
              : "Activate a model first. The drift test uses that model's own reference data."}
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
