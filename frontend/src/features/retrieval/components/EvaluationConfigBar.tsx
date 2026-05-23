import { useEffect, useState } from "react";
import { Check, KeyRound, Save } from "lucide-react";
import {
  fetchEvaluationConfig,
  getStoredEvaluationConfig,
  saveEvaluationConfig,
  storeEvaluationConfig,
  type EvaluationConfig,
} from "../api/evaluationConfigApi";

export function EvaluationConfigBar() {
  const [config, setConfig] = useState<EvaluationConfig>(() => getStoredEvaluationConfig());
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  useEffect(() => {
    let cancelled = false;
    fetchEvaluationConfig()
      .then((serverConfig) => {
        if (cancelled) return;
        const localConfig = getStoredEvaluationConfig();
        const next = {
          sessionId: localConfig.sessionId || serverConfig.sessionId || "",
          evaluationId: localConfig.evaluationId || serverConfig.evaluationId || "",
          evalServerUrl: localConfig.evalServerUrl || serverConfig.evalServerUrl || "",
        };
        setConfig(next);
        storeEvaluationConfig(next);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const updateField = (field: keyof EvaluationConfig, value: string) => {
    const next = { ...config, [field]: value };
    setConfig(next);
    storeEvaluationConfig(next);
    setStatus("idle");
  };

  const save = async () => {
    setStatus("saving");
    try {
      const saved = await saveEvaluationConfig({
        sessionId: config.sessionId.trim(),
        evaluationId: config.evaluationId.trim(),
        evalServerUrl: config.evalServerUrl.trim(),
      });
      setConfig(saved);
      setStatus("saved");
    } catch {
      setStatus("error");
    }
  };

  return (
    <section className="grid shrink-0 gap-2 rounded-2xl bg-white/95 p-2 shadow-sm ring-1 ring-slate-200 lg:grid-cols-[auto_1fr_1fr_1.2fr_auto] lg:items-center">
      <div className="flex items-center gap-2 px-2 text-xs font-black text-slate-700">
        <KeyRound size={15} />
        Evaluation
      </div>
      <input
        value={config.sessionId}
        onChange={(event) => updateField("sessionId", event.target.value)}
        placeholder="Session ID"
        className="h-9 min-w-0 rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100"
      />
      <input
        value={config.evaluationId}
        onChange={(event) => updateField("evaluationId", event.target.value)}
        placeholder="Evaluation ID"
        className="h-9 min-w-0 rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100"
      />
      <input
        value={config.evalServerUrl}
        onChange={(event) => updateField("evalServerUrl", event.target.value)}
        placeholder="Evaluation server URL"
        className="h-9 min-w-0 rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100"
      />
      <button
        onClick={save}
        className={`flex h-9 items-center justify-center gap-1.5 rounded-xl px-3 text-xs font-black transition ${
          status === "saved"
            ? "bg-emerald-100 text-emerald-700"
            : status === "error"
              ? "bg-rose-100 text-rose-700"
              : "bg-slate-900 text-white hover:bg-slate-800"
        }`}
      >
        {status === "saved" ? <Check size={14} /> : <Save size={14} />}
        {status === "saving" ? "Saving" : status === "saved" ? "Saved" : status === "error" ? "Retry" : "Save"}
      </button>
    </section>
  );
}
