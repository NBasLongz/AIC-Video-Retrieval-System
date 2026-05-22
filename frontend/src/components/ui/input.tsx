import type { InputHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm font-medium outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100",
        className,
      )}
      {...props}
    />
  );
}

