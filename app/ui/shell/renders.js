// What the steps that rewrite a finished render (upscale, interpolate) share: naming a file, waiting for its background job, swapping it under every
// scene that plays it, and remembering which steps a file has been through so two "always" modes can never feed each other forever. Pure of the page.
export const keyOf = (m) => `${m.type || "output"}/${m.subfolder || ""}/${m.filename}`;

/** Wait for a FunUI background job. `read(id)` -> {state: running | done | error, media, detail}. -> the media, or throws a sentence. `sleep` is injectable so a test need not wait. */
export async function waitJob(read, id, sleep = (ms) => new Promise((r) => setTimeout(r, ms))) {
  let lost = 0;
  for (;;) {
    await sleep(1000);
    let r;
    try { r = await read(id); } catch (err) {
      if (err.status === 404) throw new Error("ComfyUI restarted while it was working");
      if (++lost < 30) continue;                                    // a dropped reply is not a failed job
      throw new Error("lost contact with ComfyUI");
    }
    lost = 0;
    if (r.state === "error") throw new Error(r.detail || "failed inside ComfyUI");
    if (r.state === "done") return r.media;
  }
}

/** One file backs a whole chain run, so every scene render and ghost that plays `from` gets `to`. -> how many were swapped. */
export function swap(p, from, to) {
  const same = (m) => !!m && keyOf(m) === keyOf(from);
  let n = 0;
  for (const [id, r] of Object.entries(p.scene_renders || {})) if (r && same(r.media)) { p.scene_renders[id] = { ...r, media: to }; n += 1; }
  for (const list of Object.values(p.scene_variants || {})) list.forEach((t) => { if (same(t.media)) { t.media = to; n += 1; } });       // a take stepped back to must be the new file too
  p.scene_ghosts = (p.scene_ghosts || []).map((g) => (same(g.media) ? (n += 1, { ...g, media: to }) : g));
  return n;
}

/** Where each step writes its files (modules/output/render/upscale.py and modules/output/interpolate/nodes.py say the same). */
export const FOLDERS = { upscale: "funui_upscaled", interpolate: "funui_interpolated" };
const steps = new Map();                                       // file key -> the steps it has been through (kept for this page's life)
const doneOf = (m) => new Set([...(steps.get(keyOf(m)) || []), ...Object.keys(FOLDERS).filter((k) => FOLDERS[k] === m.subfolder)]);
/** Whether `media` already went through `step`. */
export const wentThrough = (media, step) => doneOf(media).has(step);
/** `to` came out of `from` by `step`: it has been through all of `from`'s steps and this one. */
export const derived = (from, to, step) => { steps.set(keyOf(to), doneOf(from).add(step)); };
