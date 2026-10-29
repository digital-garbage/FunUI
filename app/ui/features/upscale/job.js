// One upscale job from queue to result. Pure of the page.
import { waitJob } from "../../shell/renders.js";
export { keyOf, swap } from "../../shell/renders.js";
export const UPSCALED = "funui_upscaled";
export const RTX = "NVIDIA RTX Super Resolution";           // sent as the model name for that engine; the same string as modules/output/render/rtx.py NAME
export const RTX_DEFAULTS = { quality: "ULTRA", resize: "scale", scale: 2, width: 1920, height: 1080, strength: 1, tenbit: false };
const LEVEL = { LOW: "Low", MEDIUM: "Medium", HIGH: "High", ULTRA: "Ultra" };
/** "DENOISE_HIGH" -> "Denoise · High": what the SDK calls a quality level, said to a person. */
export const qualityLabel = (q) => {
  if (q === "BICUBIC") return "Bicubic (no AI)";
  const i = q.lastIndexOf("_"), level = LEVEL[q.slice(i + 1)] || q.slice(i + 1);
  if (i < 0) return `Upscale · ${LEVEL[q] || q}`;
  const kind = { DENOISE: "Denoise", DEBLUR: "Deblur", HIGHBITRATE: "High bitrate", STREAMING: "Streaming" }[q.slice(0, i)] || q.slice(0, i);
  return `${kind} · ${level}`;
};
/** Denoise and deblur clean a video at its own size: scale and target size do not apply. */
export const keepsSize = (q) => q.startsWith("DENOISE_") || q.startsWith("DEBLUR_");

/** -> the upscaled media, or throws a sentence. `sleep` is injectable so a test need not wait. */
export async function upscale(api, media, model, sleep, rtx) {
  return waitJob(api.upscaleJob, (await api.upscale(media, model, rtx)).job_id, sleep);
}
