// Centralized typed API client for the DriftTrace backend.
// All requests go through here - no scattered fetch calls in components.
//
// Base URL: in dev, Vite proxies "/api" -> backend (see vite.config.ts). In a built
// deployment, set VITE_API_BASE to the backend origin. Defaults to "/api".

import type {
  Explanation,
  HealthResponse,
  Metrics,
  ModelInfoResponse,
  ModelStatusResponse,
  PredictResponse,
  RcaLatestResponse,
  ReadyResponse,
  RunScenarioResponse,
  ScenarioName,
  UploadModelResponse,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (err) {
    throw new ApiError(
      `Network error contacting the DriftTrace API (${(err as Error).message})`,
    );
  }
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* ignore parse errors */
    }
    throw new ApiError(detail, resp.status);
  }
  return (await resp.json()) as T;
}

async function requestText(path: string): Promise<string> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) throw new ApiError(`${resp.status} ${resp.statusText}`, resp.status);
  return resp.text();
}

/**
 * POST multipart/form-data. Do NOT set Content-Type manually - the browser sets
 * it with the correct boundary. Used for model file upload.
 */
async function requestMultipart<T>(path: string, form: FormData): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE}${path}`, { method: "POST", body: form });
  } catch (err) {
    throw new ApiError(
      `Network error contacting the DriftTrace API (${(err as Error).message})`,
    );
  }
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* ignore parse errors */
    }
    throw new ApiError(detail, resp.status);
  }
  return (await resp.json()) as T;
}

/** Parse Prometheus-style text exposition into a flat name->value map. */
function parseMetrics(text: string): Metrics {
  const out: Metrics = {};
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const [name, value] = trimmed.split(/\s+/);
    if (name && value !== undefined) {
      const num = Number(value);
      if (!Number.isNaN(num)) out[name] = num;
    }
  }
  return out;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  ready: () => request<ReadyResponse>("/ready"),
  modelInfo: () => request<ModelInfoResponse>("/model-info"),
  predict: (income: number, requestId?: string) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify({ income, request_id: requestId ?? null }),
    }),
  rcaLatest: () => request<RcaLatestResponse>("/rca/latest"),
  explain: (income: number, method: "shap" | "lime") =>
    request<Explanation>("/explain", {
      method: "POST",
      body: JSON.stringify({ income, method }),
    }),
  metrics: async (): Promise<Metrics> => parseMetrics(await requestText("/metrics")),
  runScenario: (scenario: ScenarioName, n = 300, seed = 7) =>
    request<RunScenarioResponse>("/demo/run-scenario", {
      method: "POST",
      body: JSON.stringify({ scenario, n, seed }),
    }),
  // ---- Phase 5: model onboarding ----
  uploadModel: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return requestMultipart<UploadModelResponse>("/models/upload", form);
  },
  modelStatus: (modelId: string) =>
    request<ModelStatusResponse>(`/models/${encodeURIComponent(modelId)}`),
};
