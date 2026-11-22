// The helper: ask the model to act on the project. It sees a sheet of the scenes and proposes changes; each one is a card with Allow and
// Disallow, and an allowed change is made here, in the open project (so it can be undone like any other edit).
import { composer as c } from "../../composer/composer.js";
import { effFps, effFrames, genUnitId, snapFrames } from "../../shell/scenes.js";
import { addScene, split } from "../../shell/edits.js";
import { applyStory, shownStory } from "../../shell/story.js";
import { MAX_REFERENCES, sceneReferences } from "../../shell/reference_wiring.js";
import { engineOf, scrollParent } from "../../shell/chatkit.js";
import { clampBlocks } from "../../shell/blocks.js";

const NODE = "FunUIEnhancePrompt";
const log = new Map();      // project id -> [{ you, reply, status, done }]: what the helper and the person said, for as long as the page is open
let draft = "";

/** The pictures a scene works from, as ids the model can compare: ", references: ab12 cd34, aspect source: ef56". */
const short = (id) => String(id);
const refsOf = (s, fixed) => { const ids = sceneReferences(s, fixed).ids; return `${ids.length ? `, references: ${ids.map((id) => (id in (fixed || {}) ? `${short(id)} (pinned to every scene)` : short(id))).join(" ")}` : ", no references"}${s.source_image ? `, aspect source: ${short(s.source_image)}` : ""}`; };

/** The modules whose settings the helper can reach, one line (the settings themselves come from find_settings). */
const modulesLine = (ps) => { const seen = new Set(catalogOf(ps).map((x) => x.title)); return seen.size ? `\nModules with settings: ${[...seen].join(", ")}.` : ""; };
const actionsLine = (app) => {
  const list = (app.actions || []).filter((x) => !x.when || x.when()), phase = app.generate && app.generate.state ? app.generate.state().phase : "";
  return (list.length ? `\nActions you can run (run_action): ${list.map((x) => `${x.id} = ${x.label}`).join("; ")}.` : "") + (phase ? `\nGeneration right now: ${phase}.` : "");
};
const presetsLine = (list) => (list.length ? `\nPipeline starting points (apply_preset): ${list.map((x) => `${x.id} = ${x.title}`).join("; ")}.` : "");

/** The project as the model reads it: one line per scene, numbered from 1, with its length and prompt. */
export function sheetOf(project) {
  if (!project) return "";
  const fps = Number(project.frame_rate) || 25, units = (project.scenes || []).map(genUnitId), lines = (project.scenes || []).map((s, i) => `Scene ${i + 1} (${(effFrames(s, project) / effFps(s, project)).toFixed(2)} s${s.excluded ? ", left out" : ""}${refsOf(s, project.fixed_references)}${typeof s.anchor === "string" ? `, own prefix: ${s.anchor.trim() || "(none)"}` : ""}${units.filter((u) => u === units[i]).length > 1 ? `, one clip with scenes ${units.map((u, j) => (u === units[i] ? j + 1 : 0)).filter(Boolean).join(" and ")}` : ""}): ${(s.text || "").trim() || "(empty)"}`);
  const vars = (project.variables || []).map((v) => `$${v.name} = ${String(v.value).slice(0, 200)}`);
  return [`Anchor (also called prefix, put in front of every scene prompt): ${(project.anchor || "").trim() || "(empty)"}`, `Postfix (${project.postfix_enabled === false ? "switched off" : "on"}): ${(project.postfix || "").trim() || "(empty)"}`, `Negative prompt: ${(project.negative || "").trim() || "(empty)"}`,
    `Variables: ${vars.length ? "" : "(none)"}`, ...vars, `Default clip length: ${((project.num_frames_per_scene || 97) / fps).toFixed(2)} seconds at ${fps} fps.`, ...lines].join("\n");
}

/** Every setting the app has right now, for the model to look up: module, name, kind, limits and the value in force. */
export function catalogOf(ps) {
  if (!ps || !ps.activeModules) return [];
  const values = ps.currentValues();
  return ps.activeModules().flatMap((m) => Object.entries(m.settings || {}).filter(([, spec]) => ["bool", "int", "float", "enum", "text", "multiline", "string"].includes(spec.type) || !spec.type)
    .map(([name, spec]) => ({ module: m.id, title: m.title || m.id, name, label: spec.label || name, type: spec.type || "text", min: spec.min, max: spec.max, options: spec.type === "enum" ? (spec.options || []).map((o) => o.value) : undefined,
      now: values[m.id] && values[m.id][name] !== undefined ? values[m.id][name] : spec.default, hint: String(spec.hint || "").slice(0, 160) })));
}

/** Every input of every pipeline node that holds a value (not a wire), with its kind, limits, choices and the value in force. */
export async function inputsOf(app) {
  const ps = app.pipeline, slots = (ps && ps.slots && ps.slots()) || [];
  if (!slots.length || !app.api.describeNodes) return [];
  const specs = ((await app.api.describeNodes([...new Set(slots.map((s) => s.node))]).catch(() => null)) || {}).nodes || {};
  return slots.flatMap((s) => ((specs[s.node] || {}).widgets || []).filter((w) => !Array.isArray((s.inputs || {})[w.name]))
    .map((w) => ({ slot: s.id, node: s.node, group: s.group || "", name: w.name, label: w.name, type: w.type, min: w.min, max: w.max, choices: w.type === "COMBO" ? (w.choices || []).slice(0, 200) : undefined, now: (s.inputs || {})[w.name] ?? w.default })));
}

/** Make an allowed change in the project. -> a short plain line saying what was done (or why not). */
export function applyAction(project, ev, ps, api, app) {
  const a = ev.args || {}, n = (project.project.scenes || []).length;
  const secs = (pr, sc, s) => snapFrames(s * effFps(sc, pr));
  if (ev.tool === "edit_scene") {
    const at = Number(a.scene) - 1;
    if (!(at >= 0 && at < n)) return "no such scene";
    project.edit((pr) => {
      const sc = pr.scenes[at];
      if (typeof a.text === "string" && a.text.trim()) sc.text = a.text.trim();
      if (typeof a.prefix === "string") sc.anchor = a.prefix.trim();
      else if (a.project_prefix === true) delete sc.anchor;
      if (typeof a.seconds === "number" && a.seconds > 0) { sc.frames = secs(pr, sc, a.seconds); sc.frames_mode = "custom"; }
      return true;
    });
    return `Scene ${at + 1} changed`;
  }
  if (ev.tool === "add_scene") {
    project.edit((pr) => {
      const sc = addScene(pr);
      sc.text = String(a.text || "").trim();
      const first = pr.scenes[0];
      if (first && first !== sc) {                       // a new scene starts with the first scene's references and aspect source, as the person would give it
        if ((first.references || []).length) sc.references = [...first.references];
        if (first.source_image && !sc.source_image) sc.source_image = first.source_image;
      }
      const after = Number.isInteger(a.after) ? a.after : pr.scenes.length - 1;
      pr.scenes.splice(pr.scenes.indexOf(sc), 1);
      pr.scenes.splice(Math.min(Math.max(after, 0), pr.scenes.length), 0, sc);
      return true;
    });
    return "Scene added";
  }
  if (ev.tool === "copy_reference") {
    const to = Number(a.scene) - 1, from = Number(a.from_scene) - 1;
    if (!(to >= 0 && to < n && from >= 0 && from < n && to !== from)) return "no such scene";
    project.edit((pr) => { const src = pr.scenes[from], dst = pr.scenes[to]; dst.references = [...(src.references || [])]; if (src.source_image) dst.source_image = src.source_image; else delete dst.source_image; return true; });
    if (app && app.selection) app.selection.pick(project.project.scenes[to].id);        // the media bin shows the references of the picked scene
    return `Scene ${to + 1} uses the references of scene ${from + 1}`;
  }
  if (ev.tool === "run_action" && app) {
    const act = (app.actions || []).find((x) => x.id === a.action);
    if (!act || (act.when && !act.when())) return "that action is not available now";
    const ids = (a.scenes || []).map((k) => (project.project.scenes[k - 1] || {}).id).filter(Boolean);
    if (app.selection && ids.length) ids.forEach((id, i) => app.selection.pick(id, { additive: i > 0 }));
    Promise.resolve().then(() => act.run()).catch((e) => c.toast.warn({ text: `${act.label}: ${(e && e.message) || e}` }));
    return `Started: ${act.label}`;
  }
  if (ev.tool === "set_story" && api) {
    const shown = shownStory(project.project);
    return Promise.resolve(api.storySplit(String(a.text || ""))).then(({ scenes }) => {
      let did = false;
      project.edit((pr) => { did = Boolean(applyStory(pr, scenes, shown)); return did; });
      return did ? `Story rewritten (${scenes.length} scene${scenes.length === 1 ? "" : "s"})` : "the Story was not changed";
    }, (e) => `Story not changed: ${(e && e.message) || e}`);
  }
  if (ev.tool === "share_reference") {
    const from = Number(a.from_scene) - 1;
    if (!(from >= 0 && from < n)) return "no such scene";
    let given = 0;
    project.edit((pr) => {
      const src = pr.scenes[from], fx = pr.fixed_references || {};
      for (const sc of pr.scenes) {
        if (sc === src) continue;
        const plan = sceneReferences(sc, fx), add = (src.references || []).filter((id) => !plan.ids.includes(id));
        if (add.length && plan.ids.length + add.length <= MAX_REFERENCES) { sc.references = [...(sc.references || []), ...add]; given += 1; }
      }
      return given > 0;
    });
    return given ? `${given} scene${given === 1 ? "" : "s"} got the references of scene ${from + 1}` : "every scene already has them";
  }
  if (ev.tool === "cut_scene") {
    const at = Number(a.scene) - 1;
    if (!(at >= 0 && at < n)) return "no such scene";
    let did = false;
    project.edit((pr) => { const sc = pr.scenes[at]; did = Boolean(split(pr, sc.id, typeof a.at_seconds === "number" ? Math.round(a.at_seconds * effFps(sc, pr)) : undefined)); return did; });
    return did ? `Scene ${at + 1} cut in two` : "that scene cannot be cut there";
  }
  if (ev.tool === "set_cut_words" && api) {
    return Promise.resolve(api.saveStoryMarkers(a.words)).then(() => `Cut words: ${a.words.join(", ")}`, (e) => `Cut words not saved: ${(e && e.message) || e}`);
  }
  if (ev.tool === "set_project_text") {
    ({ anchor: () => project.setAnchor(a.text), postfix_enabled: () => project.setPostfixEnabled(a.text === true), postfix: () => project.setPostfix(a.text), negative: () => project.setNegative(a.text) })[a.field]?.();
    return `Changed the ${a.field}`;
  }
  if (ev.tool === "set_variable" || ev.tool === "remove_variable") {
    const list = [...(project.project.variables || [])].map((v) => ({ ...v })), at = list.findIndex((v) => v.name === a.name);
    if (ev.tool === "remove_variable") { if (at >= 0) list.splice(at, 1); } else if (at >= 0) list[at].value = a.value; else list.push({ name: a.name, value: a.value });
    project.setField("variables", list);
    return ev.tool === "remove_variable" ? `Removed $${a.name}` : `Set $${a.name}`;
  }
  if ((ev.tool === "save_shortcut" || ev.tool === "remove_shortcut") && api) {
    const done = ev.tool === "remove_shortcut" ? api.deleteShortcut(a.name) : api.saveShortcut({ name: a.name, triggers: a.triggers, replacements: a.replacements, enabled: true, category: a.category || "", sub_category: "" }, a.existing ? a.name : undefined);
    return Promise.resolve(done).then(() => (ev.tool === "remove_shortcut" ? `Deleted shortcut ${a.name}` : `Saved shortcut ${a.name}`), (e) => `Shortcut not saved: ${(e && e.message) || e}`);
  }
  if (ev.tool === "set_node_inputs" && ps) {
    const inputs = {};
    for (const x of a.changes || []) (inputs[x.slot] ||= {})[x.input] = x.value;
    return Promise.resolve(ps.save({ inputs })).then(() => `Changed ${(a.changes || []).length} pipeline input(s)`);
  }
  if (ev.tool === "apply_preset" && ps && ev.preset) {
    return Promise.resolve(ps.restore({ slots: ev.preset.slots, removed: [], unwired: {} })).then((r) => {
      if (r && r.refused && r.refused.length) return `Not changed: ${r.refused[0]}`;
      Object.entries(ev.preset.prefs || {}).forEach(([k, v]) => project.setPref(k, v));
      Object.entries(ev.preset.project || {}).forEach(([k, v]) => project.setField(k, v));
      return `Pipeline replaced with ${ev.preset.title}`;
    });
  }
  if (ev.tool === "set_setting" && ps) {
    const spec = (ps.modulesById()[a.module] || { settings: {} }).settings[a.setting];
    if (!spec) return "no such setting";
    const blocks = ps.blocks && ps.blocks(), value = spec.model_blocks && typeof a.value === "string" ? clampBlocks(a.value, blocks) : a.value;
    ps.setModuleValue(a.module, a.setting, value);
    return `${a.title || a.module}: ${a.label || a.setting} set to ${value}`;
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
  const later = (f) => (typeof requestAnimationFrame === "function" ? requestAnimationFrame(f) : setTimeout(f, 0));
  let follow = true, pin = true;          // the view follows the answer down, unless the person scrolled up to read
  const paint = () => {
    const sc = scrollParent(history.node);
    follow = pin || !sc || sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160;
    pin = false;
    const t = turns();
    history.set(t.length ? t.flatMap((x) => [c.bubble.you({ text: x.you, who: "You" }), ...(x.thinking ? [c.bubble.model({ text: x.thinking, who: "Helper (thinking)" })] : []), ...x.done.map((d) => c.hint.default({ text: d })), x.reply ? c.bubble.model({ text: x.reply, who: "Helper" }) : c.hint.default({ text: x.status || "The helper is working…" })])
      : [c.hint.default({ text: "Ask the helper to change the project. Every change is shown first and needs Allow. It uses the engine model set in Composer > Enhance." })]);
    send.setLabel ? send.setLabel(busy ? "Working…" : "Ask helper") : null;
    if (send.setDisabled) send.setDisabled(busy);
    if (follow) later(() => { const sc = scrollParent(history.node); if (sc) sc.scrollTop = sc.scrollHeight; });
  };
  async function ask() {
    const message = draft.trim(), inp = inputs(), eng = engineOf(inp);
    if (!message || busy) return;
    if (!eng) { c.toast.warn({ text: "The helper needs an engine model: set one in Composer > Enhance." }); return; }
    const turn = { you: message, thinking: "", reply: "", status: "", done: [] };
    log.set(p.project.id, [...turns(), turn]); draft = ""; box.setValue(""); busy = true; pin = true; paint();
    const shortcuts = await Promise.resolve(app.api.shortcuts ? app.api.shortcuts() : null).then((r) => ((r && r.shortcuts) || []).map((x) => ({ name: x.name, triggers: x.triggers, replacements: x.replacements, enabled: x.enabled }))).catch(() => []);
    const presets = await Promise.resolve(app.api.pipelinePresets ? app.api.pipelinePresets() : null).then((r) => (r && r.presets) || []).catch(() => []), nodeInputs = await inputsOf(app).catch(() => []);
    const cutWords = await Promise.resolve(app.api.storyMarkers ? app.api.storyMarkers() : null).then((r) => (r && r.markers) || []).catch(() => []);
    const body = { conversation: `helper:${p.project.id}`, message, notes: String(p.pref("enhance_notes", "") || ""), helper: { chat: `${p.project.id}:${Number(p.pref("enhance_chat_epoch", 0)) || 0}`, sheet: sheetOf(p.project) + (cutWords.length ? `\nCut words of the global prompt: ${cutWords.join(", ")}` : "") + modulesLine(app.pipeline) + presetsLine(presets) + actionsLine(app), scenes: (p.project.scenes || []).length, settings: catalogOf(app.pipeline), variables: (p.project.variables || []).map((v) => v.name), shortcuts, actions: (app.actions || []).filter((x) => !x.when || x.when()).map((x) => ({ id: x.id, label: x.label })), inputs: nodeInputs, presets: presets.map((x) => ({ id: x.id, title: x.title })) }, ...Object.fromEntries(PICK.filter((k) => inp[k] !== undefined).map((k) => [k, inp[k]])), engine_model: eng };
    try {
      const r = await app.api.enhancerChatStream(body, (ev) => {
        if (ev.kind === "answer") turn.reply += ev.text; else if (ev.kind === "thinking") turn.thinking += ev.text; else if (ev.kind === "status") turn.status = ev.text;
        else if (ev.kind === "action_request") (async () => {
          const yes = await askAction(ev);
          if (ev.tool === "apply_preset") ev.preset = presets.find((x) => x.id === ev.args.preset);
          turn.done.push(yes ? await applyAction(p, ev, app.pipeline, app.api, app) : `Refused: ${ev.summary}`);
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
