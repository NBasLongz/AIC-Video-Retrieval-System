import { ChevronLeft, ListVideo } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import type { RetrievalResult } from "../types/retrieval.types";

type Group = {
  videoId: string;
  frames: RetrievalResult[];
};

export function VideoList({
  groups,
  activeVideo,
  setActiveVideo,
  onOpenFrame,
  collapsed,
  setCollapsed,
}: {
  groups: Group[];
  activeVideo: string;
  setActiveVideo: (video: string) => void;
  onOpenFrame: (frame: RetrievalResult) => void;
  collapsed: boolean;
  setCollapsed: (value: boolean) => void;
}) {
  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="sticky top-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-sky-700 shadow-sm ring-1 ring-slate-200 hover:bg-sky-50"
        title="Open video list"
      >
        <ListVideo size={21} />
      </button>
    );
  }

  const totalFrames = groups.reduce((sum, group) => sum + group.frames.length, 0);

  return (
    <Card className="sticky top-4 rounded-3xl border-0 bg-white/95 shadow-sm backdrop-blur">
      <CardContent className="space-y-2.5 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-sky-100 p-2 text-sky-700">
              <ListVideo size={16} />
            </div>
            <div>
              <h2 className="text-sm font-black text-slate-950">Videos</h2>
              <p className="text-[11px] text-slate-500">Grouped matches</p>
            </div>
          </div>
          <button
            onClick={() => setCollapsed(true)}
            className="rounded-xl bg-slate-50 p-2 text-slate-500 hover:bg-sky-50 hover:text-sky-700"
            title="Collapse video list"
          >
            <ChevronLeft size={16} />
          </button>
        </div>

        <button
          onClick={() => setActiveVideo("all")}
          className={`w-full rounded-xl px-3 py-2 text-left text-xs font-black transition ${
            activeVideo === "all" ? "bg-sky-600 text-white" : "bg-slate-50 text-slate-700 hover:bg-sky-50"
          }`}
        >
          All videos / {totalFrames} matches
        </button>

        <div className="max-h-[calc(100vh-210px)] space-y-2 overflow-y-auto pr-1">
          {groups.map((group) => (
            <div key={group.videoId} className="rounded-2xl bg-slate-50 p-2">
              <button
                onClick={() => setActiveVideo(group.videoId)}
                className={`mb-2 flex w-full items-center justify-between rounded-xl px-2 py-1.5 text-left transition ${
                  activeVideo === group.videoId ? "bg-sky-600 text-white" : "hover:bg-white"
                }`}
              >
                <span className="text-xs font-black">{group.videoId}</span>
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-black ${activeVideo === group.videoId ? "bg-white/20" : "bg-white text-slate-500"}`}>
                  {group.frames.length}
                </span>
              </button>

              <div className="grid grid-cols-5 gap-1">
                {group.frames.slice(0, 5).map((frame) => (
                  <button
                    key={frame.id}
                    onClick={() => onOpenFrame(frame)}
                    className="rounded-lg bg-white px-1 py-1 text-[10px] font-bold text-slate-600 ring-1 ring-slate-200 hover:bg-amber-100 hover:text-amber-800"
                  >
                    {frame.timeLabel}
                  </button>
                ))}
                {group.frames.length > 5 && (
                  <span className="rounded-lg bg-white px-1 py-1 text-center text-[10px] font-black text-slate-400">+{group.frames.length - 5}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
