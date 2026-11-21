// The helper: ask the model to act on the project. It sees a sheet of the scenes and proposes changes; each one is a card with Allow and
// Disallow, and an allowed change is made here, in the open project (so it can be undone like any other edit).
import { composer as c } from "../../composer/composer.js";
import { addScene } from "../../shell/edits.js";
import { effFps, effFrames, snapFrames } from "../../shell/scenes.js";
import { engineOf } from "./chat.js";

const NODE = "FunUIEnhancePrompt";
const log = new Map();      // project id -> [{ you, reply, status, done }]: what the helper and the person said, for as long as the page is open
let draft = "";

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

const PICK = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_length", "context_tokens", "thinking", "greedy", "seed", "precision", "prefix_cache", "lookup_tokens", "compact"];

/** The Helper window: a message box in the middle of the screen. The model's own settings are the Enhance tab's. -> the window. */
export function openHelper(app, origin) {
  const p = app.project;
  if (!p.project) { c.toast.warn({ text: "Open a project first." }); return null; }
  let busy = false, win = null;
  const slot = () => ((app.pipeline && app.pipeline.slots()) || []).find((x) => x.node === NODE), inputs = () => (slot() || {}).inputs || {};
  const history = c.region.stack({ gap: "sm" });
  const box = c.textarea.md({ label: "Helper", rows: 3, value: draft, placeholder: "e.g. make scene 2 longer, or add a scene where the dog runs away", onInput: (v) => { draft = v; } });
  const send = c.button.md({ label: "Ask helper", tone: "primary", onClick: () => ask() });
  const turns = () => log.get(p.project.id) || [];
  const paint = () => {
    const t = turns();
    history.set(t.length ? t.flatMap((x) => [c.bubble.you({ text: x.you, who: "You" }), ...x.done.map((d) => c.hint.default({ text: d })), x.reply ? c.bubble.model({ text: x.reply, who: "Helper" }) : c.hint.default({ text: x.status || "The helper is working…" })])
      : [c.hint.default({ text: "Ask the helper to change the project: a scene's prompt or length, a new scene, the default clip length. Every change is shown first and needs Allow. It uses the engine model set in Composer > Enhance, which must support tool calls." })]);
    send.setLabel ? send.setLabel(busy ? "Working…" : "Ask helper") : null;
    if (send.setDisabled) send.setDisabled(busy);
    const sc = history.node.parentElement; if (sc) sc.scrollTop = sc.scrollHeight;
  };
  async function ask() {
    const message = draft.trim(), inp = inputs(), eng = engineOf(inp);
    if (!message || busy) return;
    if (!eng) { c.toast.warn({ text: "The helper needs an engine model: set one in Composer > Enhance." }); return; }
    const turn = { you: message, reply: "", status: "", done: [] };
    log.set(p.project.id, [...turns(), turn]); draft = ""; box.setValue(""); busy = true; paint();
    const body = { conversation: `helper:${p.project.id}`, message, notes: "", helper: { sheet: sheetOf(p.project), scenes: (p.project.scenes || []).length }, ...Object.fromEntries(PICK.filter((k) => inp[k] !== undefined).map((k) => [k, inp[k]])), engine_model: eng };
    try {
      const r = await app.api.enhancerChatStream(body, (ev) => {
        if (ev.kind === "answer") turn.reply += ev.text; else if (ev.kind === "status") turn.status = ev.text;
        else if (ev.kind === "action_request") (async () => {
          const yes = await askAction(ev);
          turn.done.push(yes ? applyAction(p, ev) : `Refused: ${ev.summary}`);
          paint();
          await app.api.enhancerEditDecision(ev.id, yes).catch(() => {});
        })();
        paint();
      });
      if (r && !r.ok) throw new Error(r.status || "the helper could not answer");
      if (r && r.reply) turn.reply = r.reply;
    } catch (e) { turn.reply = `No reply: ${(e && e.message) || e}`; }
    busy = false; paint();
  }
  win = c.modal.generic({ title: "Helper", subtitle: "acts on the project, you allow each change", size: "lg", origin: origin ? () => origin.node : undefined, body: c.region.stack({ gap: "sm", children: [history, box] }), actions: [send] });
  paint();
  return win;
}
