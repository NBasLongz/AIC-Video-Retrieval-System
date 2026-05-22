import { Captions, FileText, WandSparkles } from "lucide-react";

export function QuickAssistBar({
  ocrHint,
  setOcrHint,
  transcriptHint,
  setTranscriptHint,
  rerank,
  setRerank,
}: {
  ocrHint: string;
  setOcrHint: (value: string) => void;
  transcriptHint: string;
  setTranscriptHint: (value: string) => void;
  rerank: boolean;
  setRerank: (next: boolean | ((value: boolean) => boolean)) => void;
}) {
  return (
    <div className="mt-2 grid gap-2 xl:grid-cols-[1fr_1fr_auto]">
      <div className="relative">
        <FileText className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={15} />
        <input
          value={ocrHint}
          onChange={(event) => setOcrHint(event.target.value)}
          placeholder="OCR hint: text on sign, logo, number plate..."
          className="h-9 w-full rounded-xl border border-slate-200 bg-white pl-9 pr-3 text-xs font-semibold outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100"
        />
      </div>
      <div className="relative">
        <Captions className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={15} />
        <input
          value={transcriptHint}
          onChange={(event) => setTranscriptHint(event.target.value)}
          placeholder="Transcript hint: spoken phrase or Vietnamese keyword..."
          className="h-9 w-full rounded-xl border border-slate-200 bg-white pl-9 pr-3 text-xs font-semibold outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-100"
        />
      </div>
      <button
        onClick={() => setRerank((value) => !value)}
        className={`flex h-9 items-center justify-center gap-2 rounded-xl px-4 text-xs font-black transition ${
          rerank ? "bg-violet-100 text-violet-700" : "bg-slate-100 text-slate-500"
        }`}
      >
        <WandSparkles size={15} />
        Rerank Top-K
      </button>
    </div>
  );
}

