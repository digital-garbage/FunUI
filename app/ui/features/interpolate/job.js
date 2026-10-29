// One interpolation job from start to result. Pure of the page.
import { waitJob } from "../../shell/renders.js";

export const RTX = "NVIDIA RTX Frame Generation";           // the same string as modules/output/interpolate/framegen.py NAME
export const DEFAULTS = { mode: "MEDIUM", multiplier: 2, tenbit: false };
const MODE = { LOW: "Fast", MEDIUM: "Balanced", HIGH: "Best" };
/** "MEDIUM" -> "Balanced": what the SDK calls a generation mode, said to a person. */
export const modeLabel = (m) => MODE[m] || m;

/** -> the interpolated media, or throws a sentence. `sleep` is injectable so a test need not wait. */
export async function interpolate(api, media, opts, sleep) {
  return waitJob(api.interpolateJob, (await api.interpolate(media, opts)).job_id, sleep);
}
