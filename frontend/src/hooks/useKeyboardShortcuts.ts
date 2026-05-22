import { useEffect } from "react";

type ShortcutHandlers = {
  onSearch?: () => void;
  onSubmit?: () => void;
  onPin?: () => void;
  onClose?: () => void;
  onStepPrev?: () => void;
  onStepNext?: () => void;
};

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
      if (typing && event.key !== "Enter") return;

      if (event.key === "Enter") handlers.onSearch?.();
      if (event.key === "Escape") handlers.onClose?.();
      if (event.key.toLowerCase() === "s") handlers.onSubmit?.();
      if (event.key.toLowerCase() === "p") handlers.onPin?.();
      if (event.key === "ArrowLeft") handlers.onStepPrev?.();
      if (event.key === "ArrowRight") handlers.onStepNext?.();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}

