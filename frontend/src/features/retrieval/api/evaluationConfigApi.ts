import { apiFetch } from "@/lib/apiClient";

export type EvaluationConfig = {
  sessionId: string;
  evaluationId: string;
  evalServerUrl: string;
};

export function getStoredEvaluationConfig(): EvaluationConfig {
  return {
    sessionId: localStorage.getItem("sessionId") || "",
    evaluationId: localStorage.getItem("evaluationId") || "",
    evalServerUrl: localStorage.getItem("evalServerUrl") || "",
  };
}

export function storeEvaluationConfig(config: Partial<EvaluationConfig>) {
  if (config.sessionId !== undefined) localStorage.setItem("sessionId", config.sessionId);
  if (config.evaluationId !== undefined) localStorage.setItem("evaluationId", config.evaluationId);
  if (config.evalServerUrl !== undefined) localStorage.setItem("evalServerUrl", config.evalServerUrl);
}

export async function fetchEvaluationConfig(): Promise<EvaluationConfig> {
  return apiFetch<EvaluationConfig>("/api/evaluation-config");
}

export async function saveEvaluationConfig(config: EvaluationConfig): Promise<EvaluationConfig> {
  const saved = await apiFetch<EvaluationConfig>("/api/evaluation-config", {
    method: "POST",
    body: JSON.stringify(config),
  });
  storeEvaluationConfig(saved);
  return saved;
}
