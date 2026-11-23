// A generation error shortened for a toast: ComfyUI can hand back pages of source code, and what matters is usually at the very end.
export function brief(text, head = 160, tail = 240) {
  const t = String(text ?? "").replace(/\s+/g, " ").trim();
  return t.length > head + tail + 20 ? `${t.slice(0, head).trimEnd()} … ${t.slice(-tail).trimStart()}` : t;
}
