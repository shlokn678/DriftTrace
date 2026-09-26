import { useState } from "react";
import { api, ApiError } from "../api/client";
import type { Explanation } from "../api/types";
import { BentoCard, Button, EmptyState, ErrorState, LoadingState, MonoLabel, Pill } from "../components/primitives";
import { formatNumber } from "../lib/status";

/** SHAP (primary) / LIME (secondary) explanation via /explain, off the hot path. */
export function ExplanationPanel() {
  const [income, setIncome] = useState(4200);
  const [method, setMethod] = useState<"shap" | "lime">("shap");
  const [exp, setExp] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setExp(await api.explain(income, method));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const maxAbs = exp
    ? Math.max(...exp.attributions.map((a) => Math.abs(a.attribution)), 1e-9)
    : 1;

  return (
    <BentoCard className="col-7" major>
      <div className="card__head">
        <MonoLabel>Model Explanation</MonoLabel>
        <div className="row row-3">
          <Pill variant={method === "shap" ? "accent" : undefined}>SHAP primary</Pill>
          <Pill variant={method === "lime" ? "accent" : undefined}>LIME secondary</Pill>
        </div>
      </div>

      <div className="row row-3 wrap" style={{ marginBottom: "var(--space-4)" }}>
        <label className="stack stack-2">
          <MonoLabel>Income</MonoLabel>
          <input
            type="number"
            min={0}
            value={income}
            onChange={(e) => setIncome(Number(e.target.value))}
            className="expl-input mono-value"
            aria-label="Income for explanation"
          />
        </label>
        <div className="stack stack-2">
          <MonoLabel>Method</MonoLabel>
          <div className="row row-3">
            <Button variant={method === "shap" ? "primary" : "default"} onClick={() => setMethod("shap")}>
              SHAP
            </Button>
            <Button variant={method === "lime" ? "primary" : "default"} onClick={() => setMethod("lime")}>
              LIME
            </Button>
          </div>
        </div>
        <div className="stack stack-2" style={{ justifyContent: "flex-end" }}>
          <span aria-hidden style={{ height: 14 }} />
          <Button variant="default" onClick={run} disabled={loading}>
            Explain prediction
          </Button>
        </div>
      </div>

      {loading ? (
        <LoadingState label="Computing explanation" />
      ) : error ? (
        <ErrorState message={error} />
      ) : !exp ? (
        <EmptyState
          title="No explanation yet"
          hint="Explanations use the deployed model and run off the prediction hot path."
        />
      ) : (
        <div className="stack stack-4">
          <div className="row between wrap row-3">
            <MonoLabel>Feature contribution ({exp.method.toUpperCase()})</MonoLabel>
            <Pill variant="accent">model v{exp.model_version ?? "-"}</Pill>
          </div>
          <div className="stack stack-3">
            {exp.attributions.map((a) => {
              const pct = (Math.abs(a.attribution) / maxAbs) * 100;
              const positive = a.attribution >= 0;
              return (
                <div key={a.feature} className="stack stack-2">
                  <div className="row between">
                    <span className="mono-value" style={{ fontWeight: 600 }}>{a.feature}</span>
                    <span className="mono-value tertiary">{formatNumber(a.attribution)}</span>
                  </div>
                  <div className="expl-bar">
                    <div
                      className="expl-bar__fill"
                      style={{
                        width: `${pct}%`,
                        background: positive ? "var(--accent)" : "var(--text-tertiary)",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="expl-caveat">
            <strong>Feature contribution</strong> is not the same as{" "}
            <strong>causal root cause</strong>. {exp.caveat}
          </div>
        </div>
      )}
    </BentoCard>
  );
}
