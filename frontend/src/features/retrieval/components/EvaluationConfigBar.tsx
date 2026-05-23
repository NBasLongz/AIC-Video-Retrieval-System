import { useEffect, useState } from "react";
import { Check, KeyRound, Save, X } from "lucide-react";
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
  const [open, setOpen] = useState(false);

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
      setOpen(false);
    } catch {
      setStatus("error");
    }
  };

  const ready = Boolean(config.sessionId.trim() && config.evaluationId.trim());
  const statusLabel = status === "saved" ? "Saved" : ready ? "Ready" : "Setup";

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className={`flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-black shadow-sm ring-1 transition ${
          ready
            ? "bg-emerald-50 text-emerald-700 ring-emerald-100 hover:bg-emerald-100"
            : "bg-white text-slate-700 ring-slate-200 hover:bg-sky-50 hover:text-sky-700"
        }`}
        title="Evaluation config"
      >
        <KeyRound size={14} />
        Eval {statusLabel}
      </button>

      {open && (
        <section className="absolute right-0 top-10 z-30 w-[min(92vw,720px)] rounded-2xl bg-white p-3 shadow-xl ring-1 ring-slate-200">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs font-black text-slate-700">
              <KeyRound size={15} />
              Evaluation config
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-full p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              title="Close"
            >
              <X size={15} />
            </button>
          </div>

          <div className="grid gap-2 lg:grid-cols-[1fr_1fr_1.2fr_auto] lg:items-center">
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
          </div>
        </section>
      )}
    </div>
  );
}
