import { apiFetch } from "@/lib/apiClient";
import { getStoredEvaluationConfig, storeEvaluationConfig } from "./evaluationConfigApi";

type LoginResponse = {
  sessionId: string;
  evaluationId: string;
  evalServerUrl?: string;
};

type SubmitResponse = {
  success?: boolean;
  error?: string;
  remote_response?: unknown;
};

export async function loginEvaluation(): Promise<LoginResponse> {
  return apiFetch<LoginResponse>("/api/login", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function submitFrame(videoId: string, timeMs: number): Promise<SubmitResponse> {
  let { sessionId, evaluationId, evalServerUrl } = getStoredEvaluationConfig();
  if (!sessionId || !evaluationId) {
    const login = await loginEvaluation();
    storeEvaluationConfig({
      sessionId: login.sessionId,
      evaluationId: login.evaluationId,
      evalServerUrl: login.evalServerUrl || evalServerUrl,
    });
    ({ sessionId, evaluationId, evalServerUrl } = getStoredEvaluationConfig());
  }

  return apiFetch<SubmitResponse>("/api/submit", {
    method: "POST",
    body: JSON.stringify({
      sessionId,
      evaluationId,
      evalServerUrl,
      videoId,
      timeMs,
    }),
  });
}
