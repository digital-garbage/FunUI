// Chat: comment on the enhancer's last rewrite. The next Generate rewrites again with every comment still applying; Reset starts over.
// The comments are the project's own (editor_settings.enhance_chat); a generate hook hands them to the enhancer node for the prompt they were about.
import { engineOf, rulesOf as rulesIn, scrollParent } from "../../shell/chatkit.js";
import { composer as c } from "../../composer/composer.js";
import { effFps, effFrames, genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { RATING_PREFIX } from "../rating/choices.js";
import { sizes } from "../../shell/bin.js";
import { askEdit } from "../../shell/editrequest.js";


const NODE = "FunUIEnhancePrompt";
const COMPACT_AT = 0.8;        // the share of the context window at which the conversation is summarised (the node uses the same)

/** How long a clip of `frames` frames plays, for the VIDEO_LENGTH macro: "5.04 seconds (121 frames at 24 fps)". "" when either is unknown. */
export function lengthText(frames, fps) {
  if (!(frames > 0) || !(fps > 0)) return "";
  const seconds = frames / fps;
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)} seconds (${frames} frames at ${Number(fps.toFixed(2))} fps)`;
}

const COMMON = [[1, 1], [4, 3], [3, 2], [16, 9], [21, 9], [5, 4], [2, 1], [3, 4], [2, 3], [9, 16], [9, 21], [4, 5], [1, 2]];
/** A picture size as an aspect ratio for the ASPECT_RATIO macro: "16:9 landscape (1920 x 1080)". "" when the size is unknown. */
export function aspectText(width, height) {
  if (!(width > 0) || !(height > 0)) return "";
  const r = width / height, near = COMMON.find(([a, b]) => Math.abs(a / b / r - 1) < 0.02);
  const shape = Math.abs(r - 1) < 0.02 ? "square" : r > 1 ? "landscape" : "portrait";
  const name = near ? `${near[0]}:${near[1]}` : r > 1 ? `${r.toFixed(2)}:1` : `1:${(1 / r).toFixed(2)}`;
  return `${name} ${shape} (${width} x ${height})`;
}

/** The target aspect ratio of a scene: its aspect ratio source picture, else the project's size. */
export function aspectOf(project, scene) {
  const known = scene && scene.source_image && sizes.get(scene.source_image);
  return known ? aspectText(known[0], known[1]) : aspectText(Number(project && project.width), Number(project && project.height));
}

/** The length of the clip a scene is made in: its clips together, each cut sharing one frame. */
export function unitLength(project, scene) {
  if (!project || !scene) return "";
  const group = project.scenes.filter((s) => genUnitId(s) === genUnitId(scene));
  const root = unitRoot(project, genUnitId(scene)) || scene;
  return lengthText(group.reduce((t, s) => t + effFrames(s, project), 0) - Math.max(0, group.length - 1), effFps(root, project));
}

/** Generate hook: the comments about THIS typed prompt go to the enhancer slot; the others are said to be left out. The clip's length goes with them. */
export function chatHook({ project, scene, slots, frames }) {
  const settings = project.editor_settings || {}, chat = settings.enhance_chat, slot = (slots || []).find((s) => s.node === NODE);
  const notes = typeof settings.enhance_notes === "string" ? settings.enhance_notes.trim() : "";
  const length = lengthText(frames, effFps(scene, project)), aspect = aspectOf(project, scene);
  if (!slot || (!Array.isArray(chat) && !notes && !length && !aspect)) return null;
  // In a conversation every comment belongs to it, whatever prompt it was made on; otherwise only the ones about this prompt apply.
  const typed = (scene.text || "").trim(), all = Array.isArray(chat) ? chat : [];
  const mine = all.filter((r) => r && ((slot.inputs || {}).to_chat === true || r.original === undefined || r.original === typed)), left = all.length - mine.length;
  return { inputs: mine.length || notes || length || aspect ? { [slot.id]: { ...(mine.length ? { chat: JSON.stringify(mine) } : {}), ...(notes ? { notes } : {}), ...(length ? { video_length: length } : {}), ...(aspect ? { aspect_ratio: aspect } : {}) } } : {}, notes: left ? [`${left} Chat comment(s) are about a different prompt and were left out of this run.`] : [] };
}

/** The notes after the model's REMEMBER / FORGET lines: numbers (or exact texts) are dropped from the list as it was, then new ones are added once. */
export function applyNotes(text, add = [], drop = []) {
  const lines = String(text || "").split("\n").map((x) => x.trim()).filter(Boolean);
  const gone = new Set(drop.map((d) => { const n = Number(String(d).replace(/^#|\.$/g, "")); return Number.isInteger(n) && n >= 1 ? n - 1 : lines.findIndex((x) => x.toLowerCase() === String(d).toLowerCase()); }));
  const kept = lines.filter((_, i) => !gone.has(i));
  for (const a of add.map((x) => String(x).trim()).filter(Boolean)) if (!kept.some((x) => x.toLowerCase() === a.toLowerCase())) kept.push(a);
  return kept.join("\n");
}


/** Every rule the model may edit, as the text in force now (see shell/chatkit.js); the built-in rating line opening is the Rating feature's. */
export const rulesOf = (app, inp) => rulesIn(app, inp, RATING_PREFIX);

/** The prompt a comment is about: the picked scene's own text, as typed. */
function typedOf(app) {
  const open = app.project.project, sc = open && (open.scenes.find((s) => s.id === app.selection.focus) || open.scenes.find(isGenerative));
  const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
  return root ? (root.text || "").trim() : undefined;
}
const firstOf = (r) => Object.values(r.rewrites || {})[0];

// Every message has a chat id that only rises: the enhancer numbers its runs by its clock, and a comment takes the next number after everything it has seen,
// counted on that same clock (the app learns the difference from each reply), so the order holds whichever machine the enhancer runs on.
let skew = 0;
export function learnClock(reply) { if (reply && Number.isFinite(reply.now)) skew = reply.now - Date.now(); }
const idsOf = (...xs) => xs.flat().map((x) => (x && Number.isFinite(x.chat_id) ? x.chat_id : 0));

/** Save one comment on the last rewrite (`last`: the enhancer's latest run, or null). Shared by the Chat tab and the Comment button. */
export function addComment(app, text, last, tail = last) {
  const p = app.project, comment = String(text || "").trim(), now = Array.isArray(p.pref("enhance_chat", [])) ? p.pref("enhance_chat", []) : [];
  if (!comment) return false;
  const ok = last && last.ok, prev = now.length ? firstOf(now[now.length - 1]) : null, here = typedOf(app);
  const fresh = ok && (last.answered ?? last.after) !== prev && app.lastRun.typed !== undefined && app.lastRun.typed === here;      // a rewrite of THIS prompt that nobody has commented on yet
  const chat_id = Math.max(Date.now() + skew, 1 + Math.max(0, ...idsOf(now, last, tail)));
  p.setPref("enhance_chat", [...now, { chat_id, id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, rewrites: fresh ? { whole: last.answered ?? last.after } : {}, comment, ...(tail && tail.prompt_id ? { run: tail.prompt_id } : {}), ...(here === undefined ? {} : { original: here }) }]);
  return true;
}

export { engineOf, scrollParent };

export function chat(app, own) {
  const p = app.project, page = c.region.stack({ gap: "sm" });
  let refocus = false, last = null, all = [], live = null, writing = null, tick = 0, replying = false, stick = true, alive = true, key = "", draft = "", runs = app.lastRun.n;
  own(() => { alive = false; clearTimeout(tick); });
  const rounds = () => { const r = p.pref("enhance_chat", []); return Array.isArray(r) ? r : []; };
  const first = firstOf;
  // The conversation opens at its end (newest message and the message box); "Top ^" goes back to the start.
  const scrollTo = (end) => { const sp = scrollParent(page.node); if (sp) sp.scrollTop = end ? sp.scrollHeight : 0; };
  const later = (f) => (typeof requestAnimationFrame === "function" ? requestAnimationFrame(f) : setTimeout(f, 0));
  // While the model writes, the view follows it down, unless the person scrolled up to read.
  function draw(force) {
    const sp = scrollParent(page.node), end = !sp || sp.scrollHeight - sp.scrollTop - sp.clientHeight < 160;
    paint(force);
    if ((live || writing) && end) later(() => scrollTo(true));
  }
  function paint(force) {
    if (!alive) return;
    const here = ((app.pipeline && app.pipeline.slots()) || []).some((s) => s.node === NODE);
    const list = rounds(), ok = last && last.ok, next = JSON.stringify([p.project && p.project.id, p.pref("enhance_notes", ""), list, last && (last.answered ?? last.after), here, all.length, p.pref("enhance_chat_epoch", 0)]);
    if (!force && (next === key || app.dom.typingIn(page.node, "TEXTAREA"))) return;       // not under the caret
    key = next;
    if (!p.project) return page.set([c.hint.default({ text: "Open a project first." })]);
    // Comments would reach nothing: said here, the way Enhance says it (they stay, for a pipeline that has one).
    if (!here) return page.set([c.emptyState.default({ icon: "✎", title: "No prompt enhancer here", hint: "Chat comments go to the enhancer. Pick a preset that has one, or add the “FunUI Enhance Prompt” node in Settings ▸ Models & Pipeline." })]);
    const you = (text) => c.bubble.you({ text, who: "You" }), model = (text) => c.bubble.model({ text, who: "Model" });
    const ck = `${p.project.id}:${Number(p.pref("enhance_chat_epoch", 0)) || 0}`, convo = all.filter((r) => r.conversation === ck && r.ok);
    // "Average output speed: N tokens/s", under what the model wrote (the run's status line carries it).
    const pace = (x) => { const m = /([\d.]+) tokens\/s/.exec(String((x && x.status) || x || "")); return m && Number(m[1]) > 0 ? [c.hint.default({ text: `Average output speed: ${m[1]} tokens/sec` })] : []; };
    const thought = (r) => (r.thinking ? [c.bubble.model({ text: r.thinking, who: "Model (thinking)" })] : []);
    const noted = (r) => [...(r.remember || []).map((x) => `Noted: ${x}`), ...(r.forget || []).map((x) => `Dropped note ${x}`)].map((text) => c.hint.default({ text }));
    let bubbles;
    if (convo.length) {                      // "Send everything into chat": the whole conversation in the order of the chat ids
      const asked = new Set(convo.map((r) => r.prompt_id)), stamped = (x) => Number.isFinite(x.chat_id);
      const under = (r) => list.filter((x) => !stamped(x) && x.run === r.prompt_id).map((x) => you(x.comment));      // comments from before they had an id: under the rewrite they were about
      const rows = [...convo.map((r, i) => ({ at: Number.isFinite(r.chat_id) ? r.chat_id : -1 + i / (convo.length + 1), bubbles: [you(r.asked ?? r.before), ...thought(r), model(r.answered ?? r.after), ...pace(r), ...noted(r), ...(r.compacted ? [c.hint.default({ text: "The earlier conversation was summarised here to fit the context window." }), ...(r.summary ? [c.collapsible.default({ label: "Summary the model now works from", body: c.text.sm({ text: r.summary }) })] : [])] : []), ...under(r)] })),
        ...list.filter(stamped).map((x) => ({ at: x.chat_id, bubbles: [you(x.comment)] }))].sort((a, b) => a.at - b.at);
      bubbles = [...rows.flatMap((x) => x.bubbles), ...list.filter((x) => !stamped(x) && !asked.has(x.run)).map((x) => you(x.comment))];
    } else {
      bubbles = [...list.flatMap((r) => [first(r) ? model(first(r)) : null, you(r.comment)]),
        ...(ok ? [c.hint.default({ text: `Last rewrite of: ${String(last.asked ?? last.before ?? "").slice(0, 80)}` }), ...thought(last), model(last.answered ?? last.after), ...pace(last)] : []),
        ...(!list.length && !ok ? [c.hint.default({ text: "Generate once with the enhancer on, then comment here." })] : [])];
    }
    // An answer that is still forming: the message, the model's thinking and its words so far.
    if (live) bubbles.push(you(live.message), ...(live.thinking ? [c.bubble.model({ text: live.thinking, who: "Model (thinking)" })] : []), live.answer ? model(live.answer) : c.hint.default({ text: live.status || "The model is writing…" }), ...(live.tps ? [c.hint.default({ text: `Average output speed: ${live.tps} tokens/sec` })] : []));
    // A rewrite that is running inside a generation: what the model has written so far, as it forms.
    if (writing && !live) bubbles.push(c.hint.default({ text: `Writing: ${writing.label}` }), ...(writing.thinking ? [c.bubble.model({ text: writing.thinking, who: "Model (thinking)" })] : []), writing.answer ? model(writing.answer) : c.hint.default({ text: writing.status || "The model is writing…" }));
    bubbles = bubbles.filter(Boolean);
    const lastEntry = convo.length && convo[convo.length - 1], cx = lastEntry && lastEntry.context, cxFor = lastEntry && lastEntry.kind === "chat" ? "last chat reply" : "last rewrite";
    if (cx) bubbles.push(c.hint.default({ text: `Context (${cxFor}): about ${cx.used.toLocaleString("en-US")} of ${cx.window.toLocaleString("en-US")} tokens (${Math.round(100 * cx.used / cx.window)}%). ${cx.compact ? `Summarised at ${Math.round(100 * COMPACT_AT)}%.` : "Summarising is off."}` }));      // as of the last request
    const notesBox = c.collapsible.default({ label: `Notes (${String(p.pref("enhance_notes", "") || "").split("\n").filter((x) => x.trim()).length})`,
      body: c.region.stack({ gap: "xs", children: [c.hint.default({ text: "Standing instructions, one per line. Every rewrite and chat reply starts with them; the model can add and drop lines when you tell it something lasting." }),
        c.textarea.md({ label: "Notes", rows: 4, value: String(p.pref("enhance_notes", "") || ""), placeholder: "e.g. warm light, no rain", onCommit: (v) => { p.setPref("enhance_notes", v.trim()); draw(true); } })] }) });
    const box = c.textarea.md({ label: "Message", rows: 2, value: draft, placeholder: "Ask something, or leave a comment: warmer light, no rain", onInput: (v) => { draft = v; } });
    // A comment waits and goes with the next prompt; a message gets an answer now (needs a conversation and an encoder of its own).
    const slot = ((app.pipeline && app.pipeline.slots()) || []).find((x) => x.node === NODE), inp = (slot && slot.inputs) || {};
    const tail = convo.length ? convo[convo.length - 1] : last;
    const comment = () => { if (addComment(app, draft, last, tail)) { draft = ""; draw(true); } };
    const eng = engineOf(inp), canChat = inp.to_chat === true && Boolean(inp.encoder_file || eng);
    // The model proposes a change to one of its own rules: the person decides, an allowed text is saved into the project.
    const proposal = async (ev, sl) => {
      const yes = await askEdit(ev);
      if (yes && sl && app.pipeline) await app.pipeline.save({ inputs: { [sl.id]: { [ev.field]: ev.new } } });
      await app.api.enhancerEditDecision(ev.id, yes).catch(() => {});
    };
    const shown = () => (p.project.scenes || []).find((s) => s.id === app.selection.focus && isGenerative(s)) || (p.project.scenes || []).find(isGenerative);
    const say = async () => {
      const message = draft.trim();
      if (!message || replying) return;
      replying = true; draft = ""; draw(true);
      const pick = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "max_length", "compact", "context_tokens", "thinking", "thinking_chat", "greedy", "seed", "encoder_file", "encoder_family", "engine_model", "use_comfy", "self_edit", "precision", "prefix_cache", "lookup_tokens", "compile_decode", "chat_prompt", "notes_prompt", "compact_prompt", "edit_prompt"];
      try {
        const body = { conversation: ck, message, comments: list.map((x) => x.comment), notes: String(p.pref("enhance_notes", "") || ""), video_length: unitLength(p.project, shown()), aspect_ratio: aspectOf(p.project, shown()), ...Object.fromEntries(pick.filter((k) => inp[k] !== undefined).map((k) => [k, inp[k]])), ...(eng ? { engine_model: eng } : {}) };
        if (eng && inp.self_edit) body.rules = await rulesOf(app, inp);      // the rules the model may read and propose changes to
        let r;
        if (eng) {          // the engine writes it in front of you, piece by piece
          live = { message, thinking: "", answer: "", status: "" };
          let queued = false;
          const show = () => { if (queued) return; queued = true; later(() => { queued = false; if (live) draw(true); }); };
          try {
            r = await app.api.enhancerChatStream(body, (ev) => {
              if (ev.kind === "thinking") live.thinking += ev.text; else if (ev.kind === "answer") live.answer += ev.text; else if (ev.kind === "status") live.status = ev.text; else if (ev.kind === "end" && ev.tps) live.tps = ev.tps;
              else if (ev.kind === "edit_request") proposal(ev, slot);
              show();
            });
          } finally { live = null; }
          if (r && !r.ok) throw new Error(r.status || "the model could not answer");
        } else r = await app.api.enhancerChat(body);
        if (r && r.compacted) c.toast.warn({ text: "The conversation was getting long: the earlier turns were summarised." });
        if (r && ((r.remember || []).length || (r.forget || []).length)) p.setPref("enhance_notes", applyNotes(p.pref("enhance_notes", ""), r.remember, r.forget));
      } catch (e) { draft = message; c.toast.warn({ text: `No reply: ${(e && e.message) || e}` }); }
      refocus = app.dom.typingIn(page.node, "TEXTAREA"); replying = false; stick = true; look(true);        // forced: a person typing the next message must still see this answer arrive
    };
    page.set([c.label.section({ text: "Chat" }), c.hint.default({ text: "Send gets an answer now. Comment is kept and goes with the next prompt as part of one message. With \"Send everything into chat\" on (Enhance tab), every request and answer shows here. Reset starts over." }),
      c.region.stack({ gap: "sm", children: bubbles }), notesBox, box,
      c.toolbar.default({ items: [c.button.sm({ label: replying ? "Replying…" : "Send", tone: "primary", disabled: !canChat || replying, title: canChat ? "Send a message and get an answer now" : "Needs \"Send everything into chat\" and an Encoder of its own in the Enhance tab", onClick: say }), c.button.sm({ label: "Comment", title: "Kept for the next Generate: it goes with that prompt as part of one message", onClick: comment })], trailing: [c.button.sm({ label: "Refresh", tone: "ghost", title: "Reload the conversation from the enhancer", onClick: () => { stick = true; look().then(() => draw(true)); } }), c.button.sm({ label: "Top ^", tone: "ghost", title: "Back to the start of the conversation", onClick: () => scrollTo(false) }), c.button.sm({ label: "Reset", tone: "ghost", disabled: !list.length && !convo.length, onClick: () => { p.setPref("enhance_chat", []); p.setPref("enhance_chat_epoch", (Number(p.pref("enhance_chat_epoch", 0)) || 0) + 1); draw(true); } })] })]);
    if (refocus) { refocus = false; const t = box.node.tagName === "TEXTAREA" ? box.node : box.node.querySelector("textarea"); if (t) { t.focus(); if (t.setSelectionRange) t.setSelectionRange(t.value.length, t.value.length); } }        // the caret returns to the message being typed
  }
  const look = (force) => app.api.enhancerRuns().then((r) => { learnClock(r); all = r.runs || []; last = all.filter((x) => x.kind !== "chat").slice(-1)[0] || null; draw(force); if (stick) { stick = false; later(() => scrollTo(true)); } }).catch(() => {});
  // The engine's own writing is not in ComfyUI's log or queue, so the tab asks for it while it is open.
  const watch = () => Promise.resolve().then(() => app.api.enhancerLive()).then((r) => {
    const was = writing, now = r && r.id && !r.done ? r : null;
    if (JSON.stringify(now) !== JSON.stringify(was)) { writing = now; if (alive && !app.dom.typingIn(page.node)) { if (was && !now) { stick = true; look(); } else draw(true); } else if (alive && was && !now) { refocus = true; look(true); } }       // a finished rewrite shows even while the next message is being typed
  }).catch(() => {}).finally(() => { if (alive) { tick = setTimeout(watch, writing ? 600 : 2000); if (tick && tick.unref) tick.unref(); } });
  watch();
  own(app.on((what) => { if (what === "open" || app.lastRun.n !== runs) { runs = app.lastRun.n; look(); } else draw(); }));
  if (app.pipeline && app.pipeline.subscribe) own(app.pipeline.subscribe(() => draw()));      // a pipeline that gains or loses the enhancer
  draw(true);
  look();
  return page;
}
