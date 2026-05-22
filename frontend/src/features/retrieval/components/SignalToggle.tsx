import { cn } from "@/lib/cn";

export function SignalToggle({ checked, onChange, label }: { checked: boolean; onChange: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onChange}
      className={cn(
        "rounded-full px-2.5 py-1 text-xs font-black transition",
        checked ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-400",
      )}
    >
      {label}
    </button>
  );
}

