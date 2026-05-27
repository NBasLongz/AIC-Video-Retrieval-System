import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Pause, Pin, Play, Send, SkipBack, SkipForward, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { frameToTimestamp, formatTime, keyframeUrl, timestampToFrame } from "../utils/frameUtils";
import type { NearbyFrame, RetrievalMode, RetrievalResult } from "../types/retrieval.types";
import { NearbyFrames } from "./NearbyFrames";
import { OcrChips } from "./OcrChips";
import { ScoreBreakdown } from "./ScoreBreakdown";

function evidenceForMode(item: RetrievalResult, mode: RetrievalMode) {
  if (mode === "ocr") return item.evidence.ocr || item.evidence.text;
  if (mode === "transcript" || mode === "audio") return item.evidence.transcript || item.evidence.text;
  if (mode === "visual") return item.evidence.caption || item.evidence.text || "Visual scene matched.";
  if (item.source === "OCR") return item.evidence.ocr || item.evidence.text;
  if (item.source === "Transcript") return item.evidence.transcript || item.evidence.text;
  return item.evidence.caption || item.evidence.ocr || item.evidence.transcript || item.evidence.text || "Visual scene matched.";
}

const liveNeighborOffsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4];

export function VideoModal({
  item,
  onClose,
  onSubmit,
  onPin,
  pinned,
  submitted,
  mode,
}: {
  item: RetrievalResult | null;
  onClose: () => void;
  onSubmit: (item: RetrievalResult, frameOverride?: number) => void;
  onPin: (item: RetrievalResult) => void;
  pinned: boolean;
  submitted: boolean;
  mode: RetrievalMode;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [speed, setSpeed] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [videoTime, setVideoTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const fps = useMemo(() => Number(item?.raw.fps || 25), [item]);
  const currentTimestamp = useMemo(() => frameToTimestamp(currentFrame || item?.frame || 0, fps), [currentFrame, fps, item]);

  useEffect(() => {
    if (!item) return;
    setCurrentFrame(item.frame);
    setVideoTime(item.timestamp);
    setDuration(0);
    setIsPlaying(false);
  }, [item]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed;
  }, [speed]);

  if (!item) return null;
  const evidence = evidenceForMode(item, mode);
  const showOcrChips = mode === "ocr" || (mode === "hybrid" && item.source === "OCR");
  const rangeMax = Math.max(duration, videoTime, currentTimestamp, item.timestamp, 1);
  const progressValue = Math.min(videoTime || item.timestamp || currentTimestamp, rangeMax);
  const currentKeyframeIndex = Math.max(0, item.keyframeIndex + Math.round(currentTimestamp - item.timestamp));
  const liveNeighbors = liveNeighborOffsets
    .map((offset) => ({ offset, keyframeIndex: currentKeyframeIndex + offset }))
    .filter(({ keyframeIndex }) => keyframeIndex >= 0)
    .map(({ offset, keyframeIndex }): NearbyFrame => {
      const timestamp = Math.max(0, item.timestamp + keyframeIndex - item.keyframeIndex);
      const frame = timestampToFrame(timestamp, fps);
      return {
        id: `${item.videoId}-${keyframeIndex}-${offset}`,
        videoId: item.videoId,
        keyframeIndex,
        timestamp,
        frame,
        label: offset === 0 ? "Current" : `${offset > 0 ? "+" : ""}${offset}`,
        thumbnailUrl: keyframeUrl(item.videoId, keyframeIndex),
      };
    });

  const playVideo = async () => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = speed;
    try {
      await video.play();
      setIsPlaying(true);
    } catch {
      setIsPlaying(false);
    }
  };

  const togglePlayback = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      await playVideo();
      return;
    }
    video.pause();
    setIsPlaying(false);
  };

  const seekToFrame = (frame: number) => {
    const nextFrame = Math.max(0, frame);
    const timestamp = frameToTimestamp(nextFrame, fps);
    setCurrentFrame(nextFrame);
    setVideoTime(timestamp);
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
    }
  };

  const seekToTime = (timestamp: number) => {
    const safeTimestamp = Math.max(0, Math.min(timestamp, rangeMax));
    setVideoTime(safeTimestamp);
    setCurrentFrame(timestampToFrame(safeTimestamp, fps));
    if (videoRef.current) {
      videoRef.current.currentTime = safeTimestamp;
    }
  };

  const syncFrameFromVideo = () => {
    if (!videoRef.current) return;
    const timestamp = videoRef.current.currentTime;
    setVideoTime(timestamp);
    setCurrentFrame(timestampToFrame(timestamp, fps));
  };

  const stepFrame = (offset: number) => {
    videoRef.current?.pause();
    setIsPlaying(false);
    seekToFrame(currentFrame + offset);
  };

  const selectNeighbor = (frame: NearbyFrame) => {
    videoRef.current?.pause();
    setIsPlaying(false);
    seekToFrame(frame.frame);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/82 p-2 backdrop-blur-sm">
      <div className="flex max-h-[96vh] w-full max-w-[min(98vw,1600px)] flex-col overflow-hidden rounded-3xl bg-white shadow-2xl">
        <div className="flex shrink-0 items-center justify-between border-b border-slate-100 px-4 py-2.5">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-sky-600">Video Preview</p>
            <h2 className="text-xl font-black text-slate-950">{item.videoId}</h2>
          </div>
          <button onClick={onClose} className="rounded-full bg-slate-100 p-2 text-slate-700 hover:bg-slate-200" title="Close">
            <X size={20} />
          </button>
        </div>

        <div className="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="flex min-h-0 flex-col overflow-hidden rounded-3xl bg-slate-950">
            <div className="relative min-h-0 flex-1 bg-slate-950">
              <video
                ref={videoRef}
                src={item.videoUrl}
                className="h-full w-full object-contain"
                autoPlay
                preload="metadata"
                onClick={() => void togglePlayback()}
                onLoadedMetadata={() => {
                  if (videoRef.current && Number.isFinite(videoRef.current.duration)) {
                    setDuration(videoRef.current.duration);
                  }
                  seekToFrame(item.frame);
                  void playVideo();
                }}
                onTimeUpdate={syncFrameFromVideo}
                onSeeked={syncFrameFromVideo}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
              />
              <div className="pointer-events-none absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4 text-white">
                <div className="flex items-center justify-between text-sm font-black">
                  <span>{formatTime(currentTimestamp)}</span>
                  <span>Current frame {currentFrame}</span>
                </div>
              </div>
            </div>
            <div className="shrink-0 bg-slate-950 px-4 py-2">
              <div className="mb-2 grid grid-cols-[42px_minmax(0,1fr)_42px] items-center gap-2 text-[11px] font-black text-slate-300">
                <span>{formatTime(progressValue)}</span>
                <input
                  type="range"
                  min="0"
                  max={rangeMax}
                  step={1 / Math.max(1, fps)}
                  value={progressValue}
                  onChange={(event) => seekToTime(Number(event.target.value))}
                  className="w-full accent-sky-500"
                  aria-label="Seek video"
                />
                <span className="text-right">{formatTime(duration || rangeMax)}</span>
              </div>
              <div className="flex items-center justify-center gap-2">
              <Button
                size="icon"
                onClick={() => stepFrame(-1)}
                className="rounded-full bg-sky-600 text-white shadow-lg ring-2 ring-white/20 hover:bg-sky-500"
              >
                <SkipBack size={17} />
              </Button>
              <Button
                onClick={() => void togglePlayback()}
                className="rounded-full bg-emerald-500 px-6 text-white shadow-lg ring-2 ring-white/20 hover:bg-emerald-400"
              >
                {isPlaying ? (
                  <Pause size={17} fill="currentColor" className="mr-2" />
                ) : (
                  <Play size={17} fill="currentColor" className="mr-2" />
                )}
                {isPlaying ? "Pause" : "Play"}
              </Button>
              <Button
                size="icon"
                onClick={() => stepFrame(1)}
                className="rounded-full bg-sky-600 text-white shadow-lg ring-2 ring-white/20 hover:bg-sky-500"
              >
                <SkipForward size={17} />
              </Button>
              </div>
            </div>
            <div className="shrink-0 border-t border-white/10 bg-slate-900 px-3 py-2">
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <p className="text-xs font-black uppercase text-slate-300">Nearby keyframes</p>
                <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-black text-slate-300 ring-1 ring-white/10">same video</span>
              </div>
              <NearbyFrames frames={liveNeighbors} currentFrame={currentFrame} currentKeyframeIndex={currentKeyframeIndex} onSelect={selectNeighbor} />
            </div>
          </div>

          <aside className="min-h-0 space-y-3 overflow-y-auto pr-1">
            <div className="rounded-2xl bg-sky-50 p-4">
              <p className="text-xs font-black uppercase text-sky-600">Current frame</p>
              <p className="mt-1 text-2xl font-black text-slate-950">{currentFrame}</p>
              <p className="text-sm text-slate-600">Score {item.score.toFixed(3)} / {item.source}</p>
              <div className="mt-3">
                <ScoreBreakdown mode={mode} scores={item.scores} />
              </div>
            </div>

            <div className="rounded-2xl bg-amber-50 p-4">
              <p className="mb-2 text-xs font-black uppercase text-amber-700">Evidence</p>
              <p className="text-sm leading-6 text-slate-700">
                {evidence}
              </p>
              {showOcrChips && (
                <div className="mt-2">
                  <OcrChips matches={item.ocrMatches} text={item.evidence.ocr} />
                </div>
              )}
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
