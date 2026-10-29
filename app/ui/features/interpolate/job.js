// One interpolation job from start to result. Pure of the page; the swap and the step ledger are shared with Upscale (shell/renders.js).

export const RTX = "NVIDIA RTX Frame Generation";           // the same string as modules/output/interpolate/framegen.py NAME
export const DEFAULTS = { mode: "MEDIUM", multiplier: 2, tenbit: false };
const MODE = { LOW: "Fast", MEDIUM: "Balanced", HIGH: "Best" };
/** "MEDIUM" -> "Balanced": what the SDK calls a generation mode, said to a person. */
export const modeLabel = (m) => MODE[m] || m;

/** -> the interpolated media, or throws a sentence. `sleep` is injectable so a test need not wait. */
export async function interpolate(api, media, opts, sleep = (ms) => new Promise((r) => setTimeout(r, ms))) {
  const { job_id: id } = await api.interpolate(media, opts);
  let lost = 0;
  for (;;) {
    await sleep(1000);
    let r;
    try { r = await api.interpolateJob(id); } catch (err) {
      if (err.status === 404) throw new Error("ComfyUI restarted while it was working");
      if (++lost < 30) continue;                                    // a dropped reply is not a failed job
      throw new Error("lost contact with ComfyUI");
    }
    lost = 0;
    if (r.state === "error") throw new Error(r.detail || "failed inside ComfyUI");
    if (r.state === "done") return r.media;
  }
}
