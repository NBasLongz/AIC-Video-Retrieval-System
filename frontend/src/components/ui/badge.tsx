import type { HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn("inline-flex items-center rounded-full bg-sky-50 px-2 py-0.5 text-[11px] font-black text-sky-700", className)}
      {...props}
    />
  );
}

