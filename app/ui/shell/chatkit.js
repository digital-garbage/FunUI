// Small helpers the chat and the helper windows share.

/** The engine model a chat runs on: the one set, or the file picked from ComfyUI's list (an older project's saved encoder runs the same way). */
export const engineOf = (inp) => inp.use_comfy === true ? "" : String(inp.engine_model || "").trim() || (inp.encoder_file ? `comfy:${inp.encoder_file}` : "");

/** The nearest ancestor that scrolls (the Composer window's body). */
export function scrollParent(node) {
  if (typeof getComputedStyle !== "function") return null;
  for (let n = node.parentElement; n; n = n.parentElement) { const o = getComputedStyle(n).overflowY; if (o === "auto" || o === "scroll") return n; }
  return null;
}
