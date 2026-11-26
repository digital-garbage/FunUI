// The helper: ask the model to act on the project. It sees a sheet of the scenes and proposes changes; each one is a card with Allow and
// Disallow, and an allowed change is made here, in the open project (so it can be undone like any other edit).
import { composer as c } from "../../composer/composer.js";
import { effFps, effFrames, genUnitId, snapFrames } from "../../shell/scenes.js";
import { addScene, split } from "../../shell/edits.js";
import { applyStory, shownStory } from "../../shell/story.js";
import { MAX_REFERENCES, sceneReferences } from "../../shell/reference_wiring.js";
import { engineOf, rulesOf, scrollParent } from "../../shell/chatkit.js";
import { askEdit } from "../../shell/editrequest.js";
import { clampBlocks } from "../../shell/blocks.js";

const NODE = "FunUIEnhancePrompt";
const PHRASES = ["Pondering", "Mulling it over", "Reading the project", "Weighing the options", "Working it out", "Sketching a plan", "Thinking it through", "Checking the scenes"];
const DOING = { read_prompt: "Reading prompts", edit_prompt: "Editing prompts", find_shortcuts: "Finding shortcuts", save_shortcut: "Editing shortcuts", edit_shortcuts: "Editing shortcuts", remove_shortcut: "Editing shortcuts",
  find_settings: "Finding a setting", set_setting: "Changing a setting", find_node_inputs: "Finding a pipeline input", set_node_inputs: "Changing the pipeline", apply_preset: "Switching the pipeline",
  edit_scene: "Editing a scene", add_scene: "Adding a scene", copy_reference: "Copying references", share_reference: "Sharing references", set_clip_length: "Setting the clip length", set_story: "Rewriting the global prompt",
  cut_scene: "Cutting a scene", set_cut_words: "Setting the cut words", set_project_text: "Editing project text", set_variable: "Editing variables", remove_variable: "Editing variables", run_action: "Running an action",
  plan: "Writing the checklist", read_chat_log: "Reading the chat log", check_item: "Checking the list" };
/** The change lines of one turn with the shortcut ones added up ("Changed 20 shortcuts", "Changed 1 shortcut", ... become one "Changed 33 shortcuts"). */
export function tally(lines) {
  const out = [];
  let sum = 0, at = -1;
  for (const l of lines) {
    const m = /^Changed (\d+) shortcuts?$/.exec(l);
    if (!m) { out.push(l); continue; }
    sum += Number(m[1]);
    if (at < 0) { at = out.length; out.push(""); }
  }
  if (at >= 0) out[at] = `Changed ${sum} shortcut${sum === 1 ? "" : "s"}`;
  return out;
}

const MARK = { done: "✓", skipped: "✕", open: "○" };
const log = new Map();      // project id -> [{ you, reply, status, done }]: what the helper and the person said; finished turns are also kept in this browser, so a refresh does not lose them
const PAINT_MS = 120, KEEP = 30, LOGKEY = (id) => `funui_helper_log:${id}`;
const cut = (v, n) => (typeof v === "string" ? v.slice(0, n) : "");
function saved(id) {
  try {
    const list = JSON.parse(localStorage.getItem(LOGKEY(id)) || "[]");
    return (Array.isArray(list) ? list : []).filter((x) => x && typeof x.you === "string").map((x) => ({ you: x.you, thinking: cut(x.thinking, 20000), live: "", reply: cut(x.reply, 20000), status: cut(x.status, 300), done: [], plan: Array.isArray(x.plan) ? x.plan.slice(0, 60).map((i) => ({ text: cut(i && i.text, 300), state: cut(i && i.state, 10), note: cut(i && i.note, 200) })) : null, t0: Number(x.t0) || 0, t1: Number(x.t1) || 1, open: false, seed: Number(x.seed) || 0, stopped: false, abort: new AbortController() }));
  } catch { return []; }          // storage blocked: the history lasts until the page is closed
}
function remember(id, list) {
  try {
    if (!list.length) localStorage.removeItem(LOGKEY(id));
    else localStorage.setItem(LOGKEY(id), JSON.stringify(list.filter((t) => t.t1).slice(-KEEP).map((t) => ({ you: t.you, thinking: t.thinking, reply: t.reply, status: t.status, plan: t.plan || null, t0: t.t0, t1: t.t1, seed: t.seed }))));
  } catch { /* lasts until the page is closed */ }
}
let draft = "";
const watchers = new Map();     // project id -> windows that wait for a running turn to end (a window closed and opened again while the helper works)

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
  if (ev.tool === "edit_shortcuts" && api) {
    return (async () => {
      let did = 0;
      const known = Object.fromEntries(((await Promise.resolve(api.shortcuts ? api.shortcuts() : null).catch(() => null) || {}).shortcuts || []).map((x) => [x.name, x]));
      try {
        for (const c of a.changes || []) {
          if (c.delete) { await api.deleteShortcut(c.name); did += 1; continue; }
          const cur = known[c.name] || {};
          await api.saveShortcut({ name: c.new_name || c.name, triggers: c.triggers || cur.triggers || [], replacements: c.replacements || cur.replacements || [], enabled: cur.enabled !== false,
            category: "category" in c ? c.category : cur.category || "", sub_category: cur.sub_category || "" }, c.name);
          did += 1;
        }
      } catch (e) { return `Changed ${did} of ${(a.changes || []).length} shortcuts, then: ${(e && e.message) || e}`; }
      return `Changed ${did} shortcut${did === 1 ? "" : "s"}`;
    })();
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
export function askAction(ev, signal) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (yes) => { if (settled) return; settled = true; win.close("answered"); resolve(yes); };
    const no = c.button.md({ label: "Disallow", tone: "ghost", onClick: () => done(false) }), yes = c.button.md({ label: "Allow", tone: "primary", onClick: () => done(true) });
    const win = c.modal.generic({ title: "Allow the helper to change the project?", subtitle: ev.reason || "", size: "md", closeOnOutside: false, closeOnEsc: false,
      body: c.region.stack({ gap: "sm", children: [c.text.sm({ text: ev.summary })] }), actions: [no, yes], onClose: () => { if (!settled) { settled = true; resolve(false); } } });
    if (signal) { if (signal.aborted) done(false); else signal.addEventListener("abort", () => done(false)); }          // the request was stopped: the question goes away as a no
  });
}

const PICK = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "context_tokens", "thinking", "thinking_helper", "greedy", "seed", "precision", "prefix_cache", "lookup_tokens", "compile_decode", "compact", "thinking_budget"];

/** The Helper window: a message box in the middle of the screen. The model's own settings are the Enhance tab's. -> the window. */
export function openHelper(app, origin) {
  const p = app.project;
  if (!p.project) { c.toast.warn({ text: "Open a project first." }); return null; }
  let busy = false, win = null, active = 0;
  const slot = () => ((app.pipeline && app.pipeline.slots()) || []).find((x) => x.node === NODE), inputs = () => (slot() || {}).inputs || {};
  const history = c.region.stack({ gap: "sm" });
  const box = c.textarea.md({ label: "Helper", rows: 3, value: draft, placeholder: "e.g. make scene 2 longer, or add a scene where the dog runs away", onInput: (v) => { draft = v; } });
  const send = c.button.md({ label: "Ask helper", tone: "primary", onClick: () => ask() });
  const stopper = c.button.md({ label: "Stop", tone: "ghost", disabled: true, title: "Stop what the helper is doing", onClick: () => stopAll() });
  const turns = () => { if (!log.has(p.project.id)) log.set(p.project.id, saved(p.project.id)); return log.get(p.project.id); };
  const clear = c.button.md({ label: "Clear", tone: "ghost", title: "Forget this conversation with the helper", onClick: () => { stopAll(); log.set(p.project.id, []); remember(p.project.id, []); Promise.resolve(app.api.enhancerChatClear ? app.api.enhancerChatClear(`helper:${p.project.id}`) : null).catch(() => {}); paint(true); } });
  const later = (f) => (typeof requestAnimationFrame === "function" ? requestAnimationFrame(f) : setTimeout(f, 0));
  let tick = 0;
  // The checklist of a request sits under its message: open while the request works, folded once it is finished, and it keeps the state it ended in.
  const planKey = (x) => (x.plan || []).map((i) => i.state[0]).join("");
  const planOf = (x) => {
    const items = x.plan || [];
    if (!items.length) return [];
    const left = items.filter((i) => i.state === "open").length;
    return [c.collapsible.default({ label: `Checklist: ${items.length - left} of ${items.length} done`, open: x.planOpen !== undefined ? x.planOpen : !x.t1, onToggle: (v) => { x.planOpen = v; },
      body: c.region.stack({ gap: "xs", children: [...items.slice(0, 20).map((i, n) => c.text.sm({ text: `${MARK[i.state] || "○"} ${n + 1}. ${i.text}${i.state === "skipped" && i.note ? ` (${i.note})` : ""}` })),
        ...(items.length > 20 ? [c.hint.default({ text: `and ${items.length - 20} more` })] : [])] }) })];
  };
  // The model's thinking stays folded: a line with a changing phrase and the time since the request began, opened by a click.
  const thoughtOf = (x, working) => {
    if (!x.thinking && !x.live && !(working && !x.reply && !x.t1)) return [];
    const live = !x.t1, secs = Math.max(0, Math.round(((x.t1 || Date.now()) - x.t0) / 1000));
    const doing = x.waiting ? "Waiting for your answer" : x.doing ? DOING[x.doing] || "Working" : "";
    const label = live ? `${doing || PHRASES[(x.seed + Math.floor(secs / 7)) % PHRASES.length]}… ${secs}s` : `Thought for ${secs}s`;
    return [c.collapsible.default({ label, open: x.open, onToggle: (v) => { x.open = v; }, body: c.text.sm({ text: [x.thinking, x.live && `Written so far (not the answer):\n${x.live}`].filter(Boolean).join("\n\n") || "Nothing written yet." }) })];
  };
  let follow = true, pin = true;          // the view follows the answer down, unless the person scrolled up to read
  // A redraw between the press and the release of a click replaces the thing pressed and the click is lost (the thinking line would not open or close
  // while the answer streamed in): redraws wait until the pointer is up.
  let held = false, dirty = false, heldTimer = 0;
  const release = () => { clearTimeout(heldTimer); setTimeout(() => { held = false; if (dirty) { dirty = false; paint(); } }, 0); };
  for (const node of [history.node]) {
    node.addEventListener("pointerdown", () => { held = true; clearTimeout(heldTimer); heldTimer = setTimeout(release, 1500); });
    node.addEventListener("pointerup", release);
    node.addEventListener("pointercancel", release);
  }
  // The stream makes news for every few tokens and a redraw rebuilds the whole history: left unchecked the page spent its time drawing and the input box
  // stopped answering. A redraw happens at most about eight times a second (`now` forces it: a new request, the end of a turn).
  let lastPaint = 0, paintTimer = 0;
  const paint = (now) => {
    const wait = PAINT_MS - (Date.now() - lastPaint);
    if (now === true || wait <= 0) { clearTimeout(paintTimer); paintTimer = 0; lastPaint = Date.now(); return paintNow(); }
    if (!paintTimer) paintTimer = setTimeout(() => { paintTimer = 0; lastPaint = Date.now(); paintNow(); }, wait);
  };
  const paintNow = () => {
    if (held) { dirty = true; return; }
    const sc = scrollParent(history.node);
    follow = pin || !sc || sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160;
    pin = false;
    const t = turns();
    // a finished turn is drawn once and its elements are kept: only the turn that is still working is built again
    const draw = (x) => [c.bubble.you({ text: x.you, who: "You" }), ...planOf(x), ...thoughtOf(x, busy), ...tally(x.done).map((d) => c.hint.default({ text: d })), x.reply ? c.bubble.model({ text: x.reply, who: "Helper" }) : c.hint.default({ text: x.status || "The helper is working…" })];
    const kept = (x) => { if (!x.t1) return draw(x); const key = `${x.reply.length}|${x.status}|${x.done.length}|${x.thinking.length}|${planKey(x)}`; if (!x.kept || x.kept.key !== key) x.kept = { key, els: draw(x) }; return x.kept.els; };
    history.set(t.length ? t.flatMap(kept)
      : [c.hint.default({ text: "Ask the helper to change the project. Every change is shown first and needs Allow. It uses the engine model set in Composer > Enhance." })]);
    send.setLabel ? send.setLabel(busy ? "Interrupt and ask" : "Ask helper") : null;       // a new message while it works stops that work and is read together with it
    if (stopper.setDisabled) stopper.setDisabled(!busy);
    if (follow) later(() => { const sc = scrollParent(history.node); if (sc) sc.scrollTop = sc.scrollHeight; });
  };
  const stopTurn = (t) => { t.stopped = true; t.abort.abort(); };
  function stopAll() {
    turns().filter((t) => !t.t1).forEach(stopTurn);
    Promise.resolve(app.api.enhancerChatStop ? app.api.enhancerChatStop(`helper:${p.project.id}`) : null).catch(() => {});
  }
  async function ask() {
    const message = draft.trim(), inp = inputs(), eng = engineOf(inp);
    if (!message) return;
    if (!eng) { c.toast.warn({ text: "The helper needs an engine model: set one in Composer > Enhance." }); return; }
    const turn = { you: message, thinking: "", live: "", reply: "", status: "", done: [], t0: Date.now(), t1: 0, open: false, seed: Math.floor(Math.random() * PHRASES.length), stopped: false, abort: new AbortController() };
    turns().forEach((t) => { t.open = false; t.planOpen = false; t.kept = null; });            // a new message never leaves a thought open
    turns().filter((t) => !t.t1).forEach(stopTurn);                        // new input stops what is still working; the server then reads the old message together with this one
    log.set(p.project.id, [...turns(), turn]); draft = ""; box.setValue(""); active += 1; busy = true; pin = true; paint(true);
    if (!tick) tick = setInterval(paint, 1000);                    // the clock beside the thinking line
    const shortcuts = await Promise.resolve(app.api.shortcuts ? app.api.shortcuts() : null).then((r) => ((r && r.shortcuts) || []).map((x) => ({ name: x.name, triggers: x.triggers, replacements: x.replacements, enabled: x.enabled, category: x.category, sub_category: x.sub_category }))).catch(() => []);
    const presets = await Promise.resolve(app.api.pipelinePresets ? app.api.pipelinePresets() : null).then((r) => (r && r.presets) || []).catch(() => []), nodeInputs = await inputsOf(app).catch(() => []);
    const cutWords = await Promise.resolve(app.api.storyMarkers ? app.api.storyMarkers() : null).then((r) => (r && r.markers) || []).catch(() => []);
    const body = { conversation: `helper:${p.project.id}`, message, notes: String(p.pref("enhance_notes", "") || ""), helper: { rules: await Promise.resolve().then(() => rulesOf(app, inp)).catch(() => ({})), chat: `${p.project.id}:${Number(p.pref("enhance_chat_epoch", 0)) || 0}`, sheet: sheetOf(p.project) + (cutWords.length ? `\nCut words of the global prompt: ${cutWords.join(", ")}` : "") + modulesLine(app.pipeline) + presetsLine(presets) + actionsLine(app), scenes: (p.project.scenes || []).length, settings: catalogOf(app.pipeline), variables: (p.project.variables || []).map((v) => v.name), shortcuts, actions: (app.actions || []).filter((x) => !x.when || x.when()).map((x) => ({ id: x.id, label: x.label })), inputs: nodeInputs, presets: presets.map((x) => ({ id: x.id, title: x.title })) }, ...Object.fromEntries(PICK.filter((k) => inp[k] !== undefined).map((k) => [k, inp[k]])), temperature: Number.isFinite(Number(inp.helper_temperature)) && inp.helper_temperature !== undefined && inp.helper_temperature !== "" ? Number(inp.helper_temperature) : 0.3, max_length: Number(inp.helper_max_length) || 0, engine_model: eng };
    let r = null, err = null;
    try { r = await app.api.enhancerChatStream(body, hear(turn, presets)); } catch (e) { err = e; }
    if (err && !turn.stopped && !err.status) return resume(turn, presets);          // the connection dropped, not the helper: follow it again
    settle(turn, r, err);
  }
  // the page lost the stream (a refresh, a dropped connection, the server restarting) but the helper may still be working: follow it again, the writing so far is sent again
  async function resume(turn, presets, tries = 0) {
    turn.status = "Reconnecting…"; turn.thinking = ""; turn.live = ""; paint();
    let r = null, err = null;
    try { r = await app.api.enhancerChatAttach(`helper:${p.project.id}`, (ev) => { if (ev.kind !== "begin") hear(turn, presets)(ev); }); } catch (e) { err = e; }
    if (!err && r === null) err = new Error("the helper stopped when the connection was lost (the server may have restarted)");
    else if (err && !turn.stopped && !err.status && tries < 5) { await new Promise((ok) => setTimeout(ok, 1000 * (tries + 1))); return resume(turn, presets, tries + 1); }
    settle(turn, r, err);
  }
  // what a request tells the page, for the turn that shows it: the writing so far, and each question that waits for Allow
  const hear = (turn, presets) => (ev) => {
      if (ev.kind === "plan") turn.plan = ev.items; else if (ev.kind === "doing") { turn.doing = ev.tool || ""; if (p.pref("helper_comments", false) === true && turn.live.trim()) { turn.done.push(turn.live.trim().slice(0, 400)); turn.live = ""; } } else if (ev.kind === "answer") turn.live += ev.text; else if (ev.kind === "thinking") turn.thinking += ev.text; else if (ev.kind === "status") turn.status = ev.text;
      else if (ev.kind === "edit_request") (async () => {          // a change to one of the enhancer's own system prompts: the same Allow as in the Chat tab
        turn.waiting = true; paint();
        const yes = !turn.stopped && (await askEdit(ev, turn.abort.signal)) && !turn.stopped, sl = slot();
        turn.waiting = false;
        if (yes && sl && app.pipeline) await app.pipeline.save({ inputs: { [sl.id]: { [ev.field]: ev.new } } });
        turn.done.push(yes ? `Saved the system prompt: ${ev.field}` : `Refused: change to ${ev.field}`);
        paint();
        await app.api.enhancerEditDecision(ev.id, yes).catch(() => {});
      })();
      else if (ev.kind === "action_request") (async () => {
        turn.waiting = true; paint();
        const yes = !turn.stopped && (await askAction(ev, turn.abort.signal)) && !turn.stopped;
        turn.waiting = false;
        if (ev.tool === "apply_preset") ev.preset = presets.find((x) => x.id === ev.args.preset);
        turn.done.push(yes ? await applyAction(p, ev, app.pipeline, app.api, app) : `Refused: ${ev.summary}`);
        paint();
        await app.api.enhancerEditDecision(ev.id, yes).catch(() => {});
      })();
      paint();
  };
  // the end of a turn: the answer, a stop, or what went wrong
  const settle = (turn, r, err) => {
    if (!err && r && r.stopped) { turn.reply = ""; turn.status = "Stopped."; }
    else if (!err && r && !r.ok) err = new Error(r.status || "the helper could not answer");
    else if (!err && r && r.reply) turn.reply = r.reply;
    if (err) { if (turn.stopped) { turn.reply = ""; turn.status = "Stopped."; } else turn.reply = `No reply: ${(err && err.message) || err}`; }
    turn.open = false; turn.t1 = Date.now(); remember(p.project.id, turns()); active -= 1; busy = active > 0; if (!busy) { clearInterval(tick); tick = 0; } paint(true);
    [...(watchers.get(p.project.id) || [])].forEach((f) => f(turn));
  };
  // after a refresh the helper may still be working on the server: follow it, so its writing and its questions show again
  async function attach() {
    if (!app.api.enhancerChatAttach || turns().some((t) => !t.t1)) return;
    let turn = null, heard = null;
    const presets = await Promise.resolve(app.api.pipelinePresets ? app.api.pipelinePresets() : null).then((r) => (r && r.presets) || []).catch(() => []);
    let r = null, err = null;
    try {
      r = await app.api.enhancerChatAttach(`helper:${p.project.id}`, (ev) => {
        if (ev.kind === "begin") {
          turn = { you: String(ev.message || ""), thinking: "", live: "", reply: "", status: "", done: [], t0: Date.now() - (Number(ev.seconds) || 0) * 1000, t1: 0, open: false, seed: Math.floor(Math.random() * PHRASES.length), stopped: false, abort: new AbortController() };
          log.set(p.project.id, [...turns(), turn]); active += 1; busy = true; pin = true;
          if (!tick) tick = setInterval(paint, 1000);
          heard = hear(turn, presets);
          paint(true);
        } else if (turn) heard(ev);
      });
    } catch (e) { err = e; }
    if (turn) settle(turn, r, err);
  }
  // a turn that still works for a window that was closed: this window shows it as working (clock, Stop) and shows its end
  const adopted = new Set(turns().filter((t) => !t.t1));
  if (adopted.size) {
    active += adopted.size; busy = true; tick = setInterval(paint, 1000);
    const ws = watchers.get(p.project.id) || new Set(); watchers.set(p.project.id, ws);
    ws.add((t) => { if (!adopted.delete(t)) return; active -= 1; busy = active > 0; if (!busy) { clearInterval(tick); tick = 0; } paint(true); });
  }
  win = c.modal.generic({ title: "Helper", subtitle: "acts on the project, you allow each change", size: "lg", origin: origin ? () => origin.node : undefined, body: c.region.stack({ gap: "sm", children: [history, box] }), actions: [clear, stopper, send] });
  paint();
  attach();
  return win;
}
