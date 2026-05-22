import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Pin, Play, Send, SkipBack, SkipForward, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { frameToTimestamp, formatTime, timestampToFrame } from "../utils/frameUtils";
import type { NearbyFrame, RetrievalResult } from "../types/retrieval.types";
import { NearbyFrames } from "./NearbyFrames";
import { OcrChips } from "./OcrChips";
import { ScoreBreakdown } from "./ScoreBreakdown";

export function VideoModal({
  item,
  neighbors,
  onClose,
  onSubmit,
  onPin,
  pinned,
  submitted,
}: {
  item: RetrievalResult | null;
  neighbors: NearbyFrame[];
  onClose: () => void;
  onSubmit: (item: RetrievalResult, frameOverride?: number) => void;
  onPin: (item: RetrievalResult) => void;
  pinned: boolean;
  submitted: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [speed, setSpeed] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);

  const fps = useMemo(() => Number(item?.raw.fps || 25), [item]);
  const currentTimestamp = useMemo(() => frameToTimestamp(currentFrame, fps), [currentFrame, fps]);

  useEffect(() => {
    if (!item) return;
    setCurrentFrame(item.frame);
  }, [item]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed;
  }, [speed]);

  if (!item) return null;

  const seekToFrame = (frame: number) => {
    const nextFrame = Math.max(0, frame);
    setCurrentFrame(nextFrame);
    if (videoRef.current) {
      videoRef.current.currentTime = frameToTimestamp(nextFrame, fps);
    }
  };

  const syncFrameFromVideo = () => {
    if (!videoRef.current) return;
    setCurrentFrame(timestampToFrame(videoRef.current.currentTime, fps));
  };

  const stepFrame = (offset: number) => {
    seekToFrame(currentFrame + offset);
  };

  const selectNeighbor = (frame: NearbyFrame) => {
    seekToFrame(frame.frame);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/82 p-4 backdrop-blur-sm">
      <div className="w-full max-w-[min(96vw,1280px)] overflow-hidden rounded-3xl bg-white shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-100 p-4">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-sky-600">Video Preview</p>
            <h2 className="text-xl font-black text-slate-950">{item.videoId}</h2>
          </div>
          <button onClick={onClose} className="rounded-full bg-slate-100 p-2 text-slate-700 hover:bg-slate-200" title="Close">
            <X size={20} />
          </button>
        </div>

        <div className="grid gap-4 p-4 lg:grid-cols-[1fr_270px]">
          <div className="overflow-hidden rounded-3xl bg-slate-950">
            <div className="relative aspect-video bg-slate-950">
              <video
                ref={videoRef}
                src={item.videoUrl}
                className="h-full w-full object-contain"
                controls
                preload="metadata"
                onLoadedMetadata={() => seekToFrame(item.frame)}
                onTimeUpdate={syncFrameFromVideo}
                onSeeked={syncFrameFromVideo}
              />
              <div className="pointer-events-none absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-white">
                <div className="mb-3 flex items-center justify-center gap-2">
                  <Button size="icon" onClick={() => stepFrame(-1)} className="pointer-events-auto rounded-full bg-white text-slate-950 hover:bg-slate-100">
                    <SkipBack size={17} />
                  </Button>
                  <Button
                    onClick={() => videoRef.current?.play()}
                    className="pointer-events-auto rounded-full bg-white px-5 text-slate-950 hover:bg-slate-100"
                  >
                    <Play size={17} fill="currentColor" className="mr-2" /> Play
                  </Button>
                  <Button size="icon" onClick={() => stepFrame(1)} className="pointer-events-auto rounded-full bg-white text-slate-950 hover:bg-slate-100">
                    <SkipForward size={17} />
                  </Button>
                </div>
                <div className="flex items-center justify-between text-sm font-black">
                  <span>{formatTime(currentTimestamp)}</span>
                  <span>Current frame {currentFrame}</span>
                </div>
              </div>
            </div>
          </div>

          <aside className="space-y-3">
            <div className="rounded-2xl bg-sky-50 p-4">
              <p className="text-xs font-black uppercase text-sky-600">Current frame</p>
              <p className="mt-1 text-2xl font-black text-slate-950">{currentFrame}</p>
              <p className="text-sm text-slate-600">Score {item.score.toFixed(3)} / {item.source}</p>
              <div className="mt-3">
                <ScoreBreakdown scores={item.scores} />
              </div>
            </div>

            <div className="rounded-2xl bg-amber-50 p-4">
              <p className="mb-2 text-xs font-black uppercase text-amber-700">Evidence</p>
              <p className="text-sm leading-6 text-slate-700">
                {item.evidence.ocr || item.evidence.transcript || item.evidence.caption || item.evidence.text || "Visual scene matched."}
              </p>
              <div className="mt-2">
                <OcrChips matches={item.ocrMatches} text={item.evidence.ocr} />
              </div>
            </div>

            <div className="rounded-2xl bg-slate-50 p-4">
              <p className="mb-2 text-xs font-black uppercase text-slate-500">Speed</p>
              <div className="grid grid-cols-3 gap-2">
                {[0.5, 1, 2].map((value) => (
                  <button
                    key={value}
                    onClick={() => setSpeed(value)}
                    className={`rounded-xl px-3 py-2 text-sm font-black ${
                      speed === value ? "bg-sky-600 text-white" : "bg-white text-slate-700 ring-1 ring-slate-200"
                    }`}
                  >
                    {value}x
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-2xl bg-slate-50 p-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="text-xs font-black uppercase text-slate-500">Nearby frames</p>
                <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-black text-slate-500 ring-1 ring-slate-200">same video</span>
              </div>
              <NearbyFrames frames={neighbors} currentFrame={currentFrame} onSelect={selectNeighbor} />
            </div>

            <div className="grid grid-cols-[48px_1fr] gap-2">
              <Button
                size="icon"
                variant="outline"
                onClick={() => onPin(item)}
                className={`h-12 rounded-2xl ${pinned ? "border-amber-200 bg-amber-100 text-amber-700" : ""}`}
              >
                <Pin size={18} />
              </Button>
              <Button
                onClick={() => onSubmit({ ...item, frame: currentFrame, timestamp: currentTimestamp, timeLabel: formatTime(currentTimestamp) }, currentFrame)}
                className={`h-12 w-full rounded-2xl text-base font-black ${
                  submitted ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-100" : "bg-emerald-500 text-white hover:bg-emerald-600"
                }`}
              >
                {submitted ? <CheckCircle2 size={18} className="mr-2" /> : <Send size={18} className="mr-2" />}
                {submitted ? "Submitted" : "Submit current frame"}
              </Button>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
