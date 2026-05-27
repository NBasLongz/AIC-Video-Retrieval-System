import type { NearbyFrame } from "../types/retrieval.types";
import { keyframeUrl } from "../utils/frameUtils";

export function NearbyFrames({
  frames,
  currentFrame,
  currentKeyframeIndex,
  onSelect,
}: {
  frames: NearbyFrame[];
  currentFrame: number;
  currentKeyframeIndex?: number;
  onSelect: (frame: NearbyFrame) => void;
}) {
  return (
    <div className="grid grid-cols-9 gap-1.5">
      {frames.map((frame) => {
        const active = currentKeyframeIndex !== undefined && frame.keyframeIndex !== undefined
          ? frame.keyframeIndex === currentKeyframeIndex
          : Math.abs(frame.frame - currentFrame) <= 1;
        const thumbnailUrl = frame.thumbnailUrl || keyframeUrl(frame.videoId, frame.keyframeIndex ?? frame.frame);
        return (
          <button
            key={frame.id}
            onClick={() => onSelect(frame)}
            className={`overflow-hidden rounded-xl text-left ring-2 transition ${
              active
                ? "bg-sky-600 text-white ring-sky-400"
                : "bg-white text-slate-700 ring-slate-200 hover:bg-sky-50 hover:ring-sky-200"
            }`}
          >
            <span className="relative block aspect-video bg-slate-900">
              <img src={thumbnailUrl} alt={`${frame.videoId} frame ${frame.frame}`} className="h-full w-full object-cover" loading="lazy" />
              <span className={`absolute left-1 top-1 rounded-full px-1.5 py-0.5 text-[9px] font-black ${active ? "bg-sky-600 text-white" : "bg-slate-950/75 text-white"}`}>
                {frame.label}
              </span>
            </span>
            <span className="block px-1 py-1 text-center text-[9px] font-black">
              {frame.keyframeIndex ?? frame.frame}
            </span>
          </button>
        );
      })}
    </div>
  );
}
