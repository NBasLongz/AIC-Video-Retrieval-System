import { apiFetch } from "@/lib/apiClient";

type LoginResponse = {
  sessionId: string;
  evaluationId: string;
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
  const sessionId = localStorage.getItem("sessionId");
  const evaluationId = localStorage.getItem("evaluationId");
  if (!sessionId || !evaluationId) {
    const login = await loginEvaluation();
    localStorage.setItem("sessionId", login.sessionId);
    localStorage.setItem("evaluationId", login.evaluationId);
  }

  return apiFetch<SubmitResponse>("/api/submit", {
    method: "POST",
    body: JSON.stringify({
      sessionId: localStorage.getItem("sessionId"),
      evaluationId: localStorage.getItem("evaluationId"),
      videoId,
      timeMs,
    }),
  });
}

