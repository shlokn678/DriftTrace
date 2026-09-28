import { useCallback, useRef, useState } from "react";
import { CheckCircle2, FileUp, Loader2, XCircle } from "lucide-react";
import type { UploadModelResponse } from "../api/types";
import { api, ApiError } from "../api/client";
import { BentoCard, Button, MonoLabel, Pill } from "../components/primitives";

interface Props {
  /** Called when a supported model has been inspected and is ready to monitor. */
  onReady?: (result: UploadModelResponse) => void;
}

type Phase = "idle" | "uploading" | "done" | "error";

const ACCEPT = ".pkl,.pickle,.joblib";

/** Human-friendly label for a "missing" info key from the backend. */
const MISSING_LABEL: Record<string, string> = {
  reference_data: "Reference (baseline) data",
  dependencies: "Dependency graph",
};

function missingLabel(key: string): string {
  return MISSING_LABEL[key] ?? key.replace(/_/g, " ");
}

/**
 * Phase 5 model onboarding. The user drops or browses ONE model file; DriftTrace
 * inspects it server-side and reports what it found. Nothing is fabricated - only
 * genuinely missing information is requested.
 */
export function Onboarding({ onReady }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<UploadModelResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (file: File) => {
      setFileName(file.name);
      setPhase("uploading");
      setError(null);
      setResult(null);
      try {
        const res = await api.uploadModel(file);
        setResult(res);
        setPhase("done");
        if (res.ready_to_monitor) onReady?.(res);
      } catch (err) {
        const msg =
          err instanceof ApiError
            ? err.message
            : `Could not inspect the model (${(err as Error).message})`;
        setError(msg);
        setPhase("error");
      }
    },
    [onReady],
  );

  const onFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) void upload(file);
    },
    [upload],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      onFiles(e.dataTransfer.files);
    },
    [onFiles],
  );

  const reset = () => {
    setPhase("idle");
    setResult(null);
    setError(null);
    setFileName(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <section className="section bento" id="onboarding" aria-labelledby="onboarding-title">
      <BentoCard className="col-12" major>
        <div className="card__head">
          <MonoLabel>Model Onboarding</MonoLabel>
          <Pill variant="accent">Model-agnostic adapter</Pill>
        </div>
        <h2 id="onboarding-title" style={{ marginBottom: "var(--space-3)" }}>
          Upload a model to start monitoring
        </h2>
        <p className="muted" style={{ marginBottom: "var(--space-5)", maxWidth: 640 }}>
          Drop one model file. DriftTrace inspects it through the adapter layer, reuses your
          existing reference data and dependency graph when available, and asks only for what
          is genuinely missing. Supported formats: {ACCEPT.replace(/,/g, ", ")} (scikit-learn).
        </p>

        {phase !== "done" && (
          <div
            className={`onboard-drop${dragging ? " onboard-drop--active" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            aria-label="Upload model file: drop here or browse"
          >
            {phase === "uploading" ? (
              <Loader2 size={26} className="spin tertiary" aria-hidden />
            ) : (
              <FileUp size={26} className="tertiary" aria-hidden />
            )}
            <div className="stack stack-2 center">
              <strong style={{ color: "var(--navy)" }}>
                {phase === "uploading"
                  ? `Inspecting ${fileName ?? "model"}...`
                  : "Drop a model file or click to browse"}
              </strong>
              <MonoLabel>{ACCEPT.replace(/,/g, "  /  ")}</MonoLabel>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              hidden
              onChange={(e) => onFiles(e.target.files)}
            />
          </div>
        )}

        {phase === "error" && error && (
          <div className="onboard-result onboard-result--bad" role="alert">
            <div className="row row-3">
              <XCircle size={18} aria-hidden style={{ color: "var(--status-drift)" }} />
              <MonoLabel>Could not inspect model</MonoLabel>
            </div>
            <p className="muted" style={{ fontSize: "0.9rem" }}>
              {error}
            </p>
            <Button onClick={reset}>Try another file</Button>
          </div>
        )}

        {phase === "done" && result && (
          <InspectionResult result={result} onReset={reset} />
        )}
      </BentoCard>
    </section>
  );
}

function InspectionResult({
  result,
  onReset,
}: {
  result: UploadModelResponse;
  onReset: () => void;
}) {
  const {
    supported,
    ready_to_monitor,
    framework,
    name,
    task,
    n_features,
    supports_proba,
    features,
    reference_available,
    dependencies_available,
    missing,
    message,
  } = result;

  if (!supported) {
    return (
      <div className="onboard-result onboard-result--bad" role="alert">
        <div className="row row-3">
          <XCircle size={18} aria-hidden style={{ color: "var(--status-drift)" }} />
          <MonoLabel>Unsupported model</MonoLabel>
        </div>
        <p className="muted" style={{ fontSize: "0.9rem" }}>
          {message ??
            "DriftTrace could not recognize this file as a supported model. The MVP adapter supports scikit-learn estimators."}
        </p>
        <Button onClick={onReset}>Try another file</Button>
      </div>
    );
  }

  return (
    <div className="stack stack-5">
      <div
        className={`onboard-result ${
          ready_to_monitor ? "onboard-result--ok" : "onboard-result--warn"
        }`}
      >
        <div className="row row-3">
          {ready_to_monitor ? (
            <CheckCircle2 size={18} aria-hidden style={{ color: "var(--status-stable)" }} />
          ) : (
            <Loader2 size={18} aria-hidden className="tertiary" />
          )}
          <MonoLabel>{ready_to_monitor ? "Model ready" : "Almost ready"}</MonoLabel>
          {ready_to_monitor && (
            <Pill variant="stable" dot>
              Ready to monitor
            </Pill>
          )}
        </div>
        {message && (
          <p className="muted" style={{ fontSize: "0.9rem" }}>
            {message}
          </p>
        )}
      </div>

      <div className="bento" style={{ gap: "var(--space-4)" }}>
        <Detail col="col-3" label="Framework" value={framework ?? "-"} />
        <Detail col="col-3" label="Model" value={name ?? "-"} />
        <Detail col="col-3" label="Task" value={task ?? "-"} />
        <Detail
          col="col-3"
          label="Features"
          value={n_features != null ? String(n_features) : "-"}
        />
      </div>

      {features.length > 0 && (
        <div className="stack stack-2">
          <MonoLabel>Detected features</MonoLabel>
          <div className="row row-3 wrap">
            {features.map((f) => (
              <span key={f} className="pill">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="row row-3 wrap">
        <Pill variant={reference_available ? "stable" : "warning"} dot>
          {reference_available ? "Reference data reused" : "Reference data missing"}
        </Pill>
        <Pill variant={dependencies_available ? "stable" : "warning"} dot>
          {dependencies_available ? "Dependency graph detected" : "Dependency graph missing"}
        </Pill>
        <Pill variant={supports_proba ? "accent" : "insufficient"}>
          {supports_proba ? "Probabilities available" : "No probabilities"}
        </Pill>
      </div>

      {missing.length > 0 && (
        <div className="onboard-missing">
          <MonoLabel>Provide the missing information</MonoLabel>
          <ul className="onboard-missing__list">
            {missing.map((key) => (
              <li key={key}>
                <strong style={{ color: "var(--navy)" }}>{missingLabel(key)}</strong>
                <span className="tertiary" style={{ fontSize: "0.82rem" }}>
                  {key === "reference_data"
                    ? " - add a baseline dataset so drift can be measured against it."
                    : key === "dependencies"
                      ? " - declare a dependency graph (config/graph.yaml) to enable upstream root-cause tracing."
                      : ""}
                </span>
              </li>
            ))}
          </ul>
          <p className="tertiary" style={{ fontSize: "0.8rem" }}>
            DriftTrace only requests what it could not auto-detect. Nothing is fabricated.
          </p>
        </div>
      )}

      <div className="row row-3">
        <Button onClick={onReset}>Onboard another model</Button>
      </div>
    </div>
  );
}

function Detail({ col, label, value }: { col: string; label: string; value: string }) {
  return (
    <div className={col}>
      <MonoLabel>{label}</MonoLabel>
      <div
        className="mono-value"
        style={{ marginTop: 6, fontWeight: 600, color: "var(--navy)" }}
      >
        {value}
      </div>
    </div>
  );
}
