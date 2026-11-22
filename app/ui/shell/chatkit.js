// Small helpers the chat and the helper windows share.
import { RULE_LABELS } from "./editrequest.js";

/** The engine model a chat runs on: the one set, or the file picked from ComfyUI's list (an older project's saved encoder runs the same way). */
export const engineOf = (inp) => inp.use_comfy === true ? "" : String(inp.engine_model || "").trim() || (inp.encoder_file ? `comfy:${inp.encoder_file}` : "");

/** The nearest ancestor that scrolls (the Composer window's body). */
export function scrollParent(node) {
  if (typeof getComputedStyle !== "function") return null;
  for (let n = node.parentElement; n; n = n.parentElement) { const o = getComputedStyle(n).overflowY; if (o === "auto" || o === "scroll") return n; }
  return null;
}

/** Every rule the model may edit, as the text in force now: the person's own, else the built-in one. */
export async function rulesOf(app, inp, ratingPrefix = "") {
  const d = (await app.api.enhancerDefaults().catch(() => null)) || {}, style = inp.style || "default";
  const built = { ...(d.prompts || {}), instructions: (d.styles || {})[style] || d.instructions || "", reference_intro: d.reference_intro || "", rating_prefix: ratingPrefix };
  return Object.fromEntries(Object.keys(RULE_LABELS).map((k) => [k, String(inp[k] || built[k] || "")]).filter(([, v]) => v));
}

