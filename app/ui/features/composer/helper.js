// The helper: ask the model to act on the project. It sees a sheet of the scenes and proposes changes; each one is a card with Allow and
// Disallow, and an allowed change is made here, in the open project (so it can be undone like any other edit).
import { composer as c } from "../../composer/composer.js";
import { addScene } from "../../shell/edits.js";
import { effFps, effFrames, snapFrames } from "../../shell/scenes.js";

/** The project as the model reads it: one line per scene, numbered from 1, with its length and prompt. */
export function sheetOf(project) {
  if (!project) return "";
  const fps = Number(project.frame_rate) || 25, lines = (project.scenes || []).map((s, i) => `Scene ${i + 1} (${(effFrames(s, project) / effFps(s, project)).toFixed(2)} s${s.excluded ? ", left out" : ""}): ${(s.text || "").trim() || "(empty)"}`);
  return [`Default clip length: ${((project.num_frames_per_scene || 97) / fps).toFixed(2)} seconds at ${fps} fps.`, ...lines].join("\n");
}

/** Make an allowed change in the project. -> a short plain line saying what was done (or why not). */
export function applyAction(project, ev) {
  const a = ev.args || {}, n = (project.project.scenes || []).length;
  const secs = (pr, sc, s) => snapFrames(s * effFps(sc, pr));
  if (ev.tool === "edit_scene") {
    const at = Number(a.scene) - 1;
    if (!(at >= 0 && at < n)) return "no such scene";
    project.edit((pr) => {
      const sc = pr.scenes[at];
      if (typeof a.text === "string" && a.text.trim()) sc.text = a.text.trim();
      if (typeof a.seconds === "number" && a.seconds > 0) { sc.frames = secs(pr, sc, a.seconds); sc.frames_mode = "custom"; }
      return true;
    });
    return `Scene ${at + 1} changed`;
  }
  if (ev.tool === "add_scene") {
    project.edit((pr) => {
      const sc = addScene(pr);
      sc.text = String(a.text || "").trim();
      const after = Number.isInteger(a.after) ? a.after : pr.scenes.length - 1;
      pr.scenes.splice(pr.scenes.indexOf(sc), 1);
      pr.scenes.splice(Math.min(Math.max(after, 0), pr.scenes.length), 0, sc);
      return true;
    });
    return "Scene added";
  }
  if (ev.tool === "set_clip_length" && typeof a.seconds === "number" && a.seconds > 0) {
    project.edit((pr) => { pr.num_frames_per_scene = snapFrames(a.seconds * (Number(pr.frame_rate) || 25)); return true; });
    return "Default clip length changed";
  }
  return "unknown change";
}

/** Ask whether the helper may make a change. `ev`: { summary, reason }. -> Promise<boolean>; closing the window means no. */
export function askAction(ev) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (yes) => { if (settled) return; settled = true; win.close("answered"); resolve(yes); };
    const no = c.button.md({ label: "Disallow", tone: "ghost", onClick: () => done(false) }), yes = c.button.md({ label: "Allow", tone: "primary", onClick: () => done(true) });
    const win = c.modal.generic({ title: "Allow the helper to change the project?", subtitle: ev.reason || "", size: "md", closeOnOutside: false, closeOnEsc: false,
      body: c.region.stack({ gap: "sm", children: [c.text.sm({ text: ev.summary })] }), actions: [no, yes], onClose: () => { if (!settled) { settled = true; resolve(false); } } });
  });
}
