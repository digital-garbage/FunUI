// What the steps that rewrite a finished render (upscale, interpolate) share: naming a file, waiting for its background job, swapping it under every
// scene that plays it, and remembering which steps a file has been through so two "always" modes can never feed each other forever. Pure of the page.
export const keyOf = (m) => `${m.type || "output"}/${m.subfolder || ""}/${m.filename}`;

const active = new Map();                                      // id -> its stop call, for the Upscale / Interpolate jobs this page is waiting for
const asked = new Set();                                       // ids Stop was pressed for: whatever ends them afterwards is a stop
const listeners = [];
const changed = () => { for (const fn of listeners) { try { fn(); } catch { /* one listener must not stop the others */ } } };
/** How many jobs the Stop button can stop right now. */
export const jobsRunning = () => active.size;
/** -> a function that stops listening. */
export const onJobs = (fn) => { listeners.push(fn); return () => { const i = listeners.indexOf(fn); if (i >= 0) listeners.splice(i, 1); }; };
/** The Stop button: ask every job this page is waiting for to stop, each through its own module's route. */
export const stopJobs = () => Promise.all([...active].map(async ([id, stop]) => {
  try { const r = await stop(id); if (r && r.stopped) asked.add(id); } catch { /* the request failed: the job runs on, and a real failure later is still a failure */ }
}));
/** What waitJob throws for a job that was stopped: not a failure, so callers say "stopped" and not "failed". */
export class JobStopped extends Error { constructor() { super("Stopped."); this.stopped = true; } }

/** Wait for a FunUI background job. `read(id)` -> {state: running | done | error | stopped, media, detail}. -> the media, or throws a sentence (a JobStopped when it was stopped). `sleep` is injectable so a test need not wait. */
export async function waitJob(read, id, sleep = (ms) => new Promise((r) => setTimeout(r, ms)), stop = () => {}) {
  let lost = 0;
  active.set(id, stop); changed();
  try {
    for (;;) {
      await sleep(1000);
      let r;
      try { r = await read(id); } catch (err) {
        if (err.status === 404) throw new Error("ComfyUI restarted while it was working");
        if (++lost < 30) continue;                                    // a dropped reply is not a failed job
        throw new Error("lost contact with ComfyUI");
      }
      lost = 0;
      if (r.state === "stopped" || (r.state === "error" && asked.has(id))) throw new JobStopped();       // a chunk that failed while Stop was landing is still a stop
      if (r.state === "error") throw new Error(r.detail || "failed inside ComfyUI");
      if (r.state === "done") return r.media;
    }
  } finally { active.delete(id); asked.delete(id); changed(); }
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
const gaveUpWatchers = [];
/** A step that was due on `media` (an "always" mode) could not run: whoever waited for it may go on without it. */
export const gaveUp = (media, step) => { for (const fn of gaveUpWatchers) fn(media, step); };
export const onGaveUp = (fn) => { gaveUpWatchers.push(fn); };

/** Every render file a project names (the one on each clip and every take), once each. */
export function renderFiles(p) {
  const seen = new Map();
  const add = (m) => { if (m && m.filename && !seen.has(keyOf(m))) seen.set(keyOf(m), { filename: m.filename, subfolder: m.subfolder || "", type: m.type || "output" }); };
  Object.values(p.scene_renders || {}).forEach((r) => add(r && r.media));
  Object.values(p.scene_variants || {}).forEach((list) => (list || []).forEach((t) => add(t && t.media)));
  return [...seen.values()];
}

const MAX_PAST_RATINGS = 5000;

/** Forget every render and take whose file is gone (`gone`: a Set of keyOf names). A clip left without a render is "not made yet" again.
 *  A rating is about the prompt it was given to, not about the file: each one moves to the project's `past_ratings`
 *  ({rating, rated: {text, at}, promptId}), where what learns from ratings still reads it. -> true when something was forgotten. */
export function forgetGone(p, gone) {
  let hit = false;
  const dead = (m) => Boolean(m && m.filename && gone.has(keyOf(m)));
  const keep = (rating, rated, text, promptId) => {
    if (!rating) return;
    const about = { ...(rated || {}) };
    if (typeof about.text !== "string") about.text = text || "";
    (p.past_ratings ||= []).push({ rating, rated: about, ...(promptId ? { promptId } : {}) });
  };
  const onClip = new Map();                 // scene id -> the render that was on it: its take holds the same rating, kept once
  for (const [id, r] of Object.entries(p.scene_renders || {})) {
    if (!dead(r && r.media)) continue;
    const sc = (p.scenes || []).find((s) => s.id === id);
    onClip.set(id, r);
    if (sc && !(sc.cut_offset_frames > 0)) keep(sc.rating, sc.rated, r.text ?? sc.text, r.promptId);       // a cut half shares its unit's rating: counted on the first half only
    if (sc) Object.assign(sc, { rating: "", rated: {} });
    delete p.scene_renders[id];
    hit = true;
  }
  for (const [id, list] of Object.entries(p.scene_variants || {})) {
    const was = onClip.get(id);
    const same = (t) => was && (was.promptId ? t.promptId === was.promptId : (t.media || {}).filename === (was.media || {}).filename);
    const left = [];
    for (const t of list || []) {
      if (!dead(t && t.media)) { left.push(t); continue; }
      if (!same(t)) keep(t.rating, t.rated, t.text, t.promptId);
    }
    if (left.length === (list || []).length) continue;
    if (left.length) p.scene_variants[id] = left; else delete p.scene_variants[id];
    hit = true;
  }
  if (p.past_ratings && p.past_ratings.length > MAX_PAST_RATINGS) p.past_ratings.splice(0, p.past_ratings.length - MAX_PAST_RATINGS);
  return hit;
}
