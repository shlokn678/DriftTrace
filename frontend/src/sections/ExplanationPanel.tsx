import { useEffect, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { api, ApiError } from "../api/client";
import type { Explanation, ExplanationFactor } from "../api/types";
import {
  BentoCard,
  Button,
  EmptyState,
  ErrorState,
  LoadingState,
  MonoLabel,
  Pill,
} from "../components/primitives";
import { formatNumber } from "../lib/status";

const STRENGTH_LABEL: Record<string, string> = {
  strong: "strong influence",
  moderate: "moderate influence",
  small: "small influence",
};

function factorText(f: ExplanationFactor): string {
  const opposing = f.direction === "away_from_prediction";
  const strength = STRENGTH_LABEL[f.strength] ?? "influence";
  return opposing ? `${f.label} — ${f.strength} opposing influence` : `${f.label} — ${strength}`;
}

interface Props {
  /** The active model's feature names (dynamic). Empty when no model is active. */
  features: string[];
}

/** SHAP (primary) / LIME (secondary) explanation via /explain, off the hot path.
 * Inputs are generated dynamically from the active model's feature names. */
export function ExplanationPanel({ features }: Props) {
  const [values, setValues] = useState<Record<string, number>>({});
  const [method, setMethod] = useState<"shap" | "lime">("shap");
  const [exp, setExp] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTech, setShowTech] = useState(false);

  // Seed one numeric input per feature (default 0) when the active model changes.
  useEffect(() => {
    setValues((prev) => {
      const next: Record<string, number> = {};
      for (const f of features) next[f] = prev[f] ?? 0;
      return next;
    });
    setExp(null);
  }, [features]);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setExp(await api.explain(values, method));
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

      {features.length === 0 ? (
        <EmptyState
          title="No active model"
          hint="Activate a model to explain a prediction over its features."
        />
      ) : (
        <>
          <div className="row row-3 wrap" style={{ marginBottom: "var(--space-4)" }}>
            {features.map((f) => (
              <label key={f} className="stack stack-2">
                <MonoLabel>{f}</MonoLabel>
                <input
                  type="number"
                  value={values[f] ?? 0}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [f]: Number(e.target.value) }))
                  }
                  className="expl-input mono-value"
                  aria-label={`${f} value for explanation`}
                />
              </label>
            ))}
          </div>
          <div className="row row-3 wrap" style={{ marginBottom: "var(--space-4)" }}>
            <div className="stack stack-2">
              <MonoLabel>Method</MonoLabel>
              <div className="row row-3">
                <Button
                  variant={method === "shap" ? "primary" : "default"}
                  onClick={() => setMethod("shap")}
                >
                  SHAP
                </Button>
                <Button
                  variant={method === "lime" ? "primary" : "default"}
                  onClick={() => setMethod("lime")}
                >
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
              hint="Explanations use the active model and run off the prediction hot path."
            />
          ) : (
            <div className="stack stack-4">
              {/* Natural-language interpretation first (the primary content). */}
              {exp.interpretation ? (
                <div className="stack stack-4">
                  <h3 style={{ marginBottom: 0 }}>Why did the model make this prediction?</h3>

                  <div className="bento" style={{ gap: "var(--space-4)" }}>
                    <div className="col-6 stack stack-2">
                      <MonoLabel>Prediction</MonoLabel>
                      <span className="mono-value" style={{ fontWeight: 700, color: "var(--navy)" }}>
                        {exp.interpretation.prediction_label}
                      </span>
                    </div>
                    <div className="col-6 stack stack-2">
                      <MonoLabel>Explanation method</MonoLabel>
                      <span className="mono-value" style={{ fontWeight: 700, color: "var(--navy)" }}>
                        {exp.interpretation.method}
                      </span>
                      <span className="tertiary" style={{ fontSize: "0.78rem" }}>
                        {exp.interpretation.scope_label}
                      </span>
                    </div>
                  </div>

                  <div className="stack stack-2">
                    <MonoLabel>Summary</MonoLabel>
                    <p className="muted" style={{ lineHeight: 1.55 }}>
                      {exp.interpretation.summary}
                    </p>
                  </div>

                  {exp.interpretation.supporting_factors.length > 0 && (
                    <div className="stack stack-2">
                      <MonoLabel>Main supporting factors</MonoLabel>
                      <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
                        {exp.interpretation.supporting_factors.map((f) => (
                          <li key={f.feature} style={{ color: "var(--navy)" }}>
                            {factorText(f)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {exp.interpretation.opposing_factors.length > 0 && (
                    <div className="stack stack-2">
                      <MonoLabel>Factors pushing the other way</MonoLabel>
                      <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
                        {exp.interpretation.opposing_factors.map((f) => (
                          <li key={f.feature} className="tertiary">
                            {factorText(f)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {exp.interpretation.other_note && (
                    <p className="tertiary" style={{ fontSize: "0.82rem" }}>
                      {exp.interpretation.other_note}
                    </p>
                  )}
                </div>
              ) : (
                <div className="row between wrap row-3">
                  <MonoLabel>Feature contribution ({exp.method.toUpperCase()})</MonoLabel>
                  <Pill variant="accent">model {exp.model_version ?? "-"}</Pill>
                </div>
              )}

              <div className="expl-caveat">
                <strong>Feature contribution</strong> is not the same as{" "}
                <strong>causal root cause</strong>. {exp.caveat}
              </div>

              {/* Raw SHAP/LIME values behind a details toggle. */}
              <div className="stack stack-3">
                <Button variant="ghost" onClick={() => setShowTech((v) => !v)}>
                  {showTech ? (
                    <>
                      <ChevronUp size={16} aria-hidden /> Hide technical{" "}
                      {exp.method.toUpperCase()} values
                    </>
                  ) : (
                    <>
                      <ChevronDown size={16} aria-hidden /> View technical{" "}
                      {exp.method.toUpperCase()} values
                    </>
                  )}
                </Button>
                {showTech && (
                  <div className="stack stack-3">
                    <div className="row between wrap row-3">
                      <MonoLabel>Feature contribution ({exp.method.toUpperCase()})</MonoLabel>
                      <Pill variant="accent">model {exp.model_version ?? "-"}</Pill>
                    </div>
                    {exp.attributions
                      .slice()
                      .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
                      .map((a) => {
                        const pct = (Math.abs(a.attribution) / maxAbs) * 100;
                        const positive = a.attribution >= 0;
                        return (
                          <div key={a.feature} className="stack stack-2">
                            <div className="row between">
                              <span className="mono-value" style={{ fontWeight: 600 }}>
                                {a.feature}
                              </span>
                              <span className="mono-value tertiary">
                                {formatNumber(a.attribution)}
                              </span>
                            </div>
                            <div className="expl-bar">
                              <div
                                className="expl-bar__fill"
                                style={{
                                  width: `${pct}%`,
                                  background: positive
                                    ? "var(--accent)"
                                    : "var(--text-tertiary)",
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </BentoCard>
  );
}
