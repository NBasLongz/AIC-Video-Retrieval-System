import type { InputHTMLAttributes } from "react";

export function Slider(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input type="range" className="accent-sky-600" {...props} />;
}

