import { retrievalModes } from "../constants/retrieval.constants";
import type { RetrievalMode } from "../types/retrieval.types";
import { cn } from "@/lib/cn";

export function ModeTabs({ mode, setMode }: { mode: RetrievalMode; setMode: (mode: RetrievalMode) => void }) {
  return (
    <div className="flex flex-wrap justify-end gap-1.5">
      {retrievalModes.map((item) => {
        const Icon = item.icon;
        const active = mode === item.id;
        return (
          <button
            key={item.id}
            onClick={() => setMode(item.id)}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-black transition",
              active
                ? "bg-sky-600 text-white shadow-md shadow-sky-200"
                : "bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-sky-50 hover:text-sky-700",
            )}
          >
            <Icon size={14} />
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

