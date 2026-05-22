export function OcrChips({ matches, text }: { matches: string[]; text?: string }) {
  const chips = matches.length ? matches : text ? [text.slice(0, 48)] : [];
  if (!chips.length) return null;

  return (
    <div className="flex flex-wrap gap-1">
      {chips.map((chip) => (
        <span key={chip} className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-black text-amber-700">
          OCR: {chip}
        </span>
      ))}
    </div>
  );
}

