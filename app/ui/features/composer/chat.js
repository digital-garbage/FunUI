// Chat: comment on the enhancer's last rewrite. The next Generate rewrites again with every comment still applying; Reset starts over.
// The comments are the project's own (editor_settings.enhance_chat); a generate hook hands them to the enhancer node for the prompt they were about.
import { composer as c } from "../../composer/composer.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";

const NODE = "FunUIEnhancePrompt";
const COMPACT_AT = 0.8;        // the share of the context window at which the conversation is summarised (the node uses the same)

/** Generate hook: the comments about THIS typed prompt go to the enhancer slot; the others are said to be left out. */
export function chatHook({ project, scene, slots }) {
  const settings = project.editor_settings || {}, chat = settings.enhance_chat, slot = (slots || []).find((s) => s.node === NODE);
  const notes = typeof settings.enhance_notes === "string" ? settings.enhance_notes.trim() : "";
  if (!slot || (!Array.isArray(chat) && !notes)) return null;
  // In a conversation every comment belongs to it, whatever prompt it was made on; otherwise only the ones about this prompt apply.
  const typed = (scene.text || "").trim(), all = Array.isArray(chat) ? chat : [];
  const mine = all.filter((r) => r && ((slot.inputs || {}).to_chat === true || r.original === undefined || r.original === typed)), left = all.length - mine.length;
  return { inputs: mine.length || notes ? { [slot.id]: { ...(mine.length ? { chat: JSON.stringify(mine) } : {}), ...(notes ? { notes } : {}) } } : {}, notes: left ? [`${left} Chat comment(s) are about a different prompt and were left out of this run.`] : [] };
}

/** The notes after the model's REMEMBER / FORGET lines: numbers (or exact texts) are dropped from the list as it was, then new ones are added once. */
export function applyNotes(text, add = [], drop = []) {
  const lines = String(text || "").split("\n").map((x) => x.trim()).filter(Boolean);
  const gone = new Set(drop.map((d) => { const n = Number(String(d).replace(/^#|\.$/g, "")); return Number.isInteger(n) && n >= 1 ? n - 1 : lines.findIndex((x) => x.toLowerCase() === String(d).toLowerCase()); }));
  const kept = lines.filter((_, i) => !gone.has(i));
  for (const a of add.map((x) => String(x).trim()).filter(Boolean)) if (!kept.some((x) => x.toLowerCase() === a.toLowerCase())) kept.push(a);
  return kept.join("\n");
}

/** The prompt a comment is about: the picked scene's own text, as typed. */
function typedOf(app) {
  const open = app.project.project, sc = open && (open.scenes.find((s) => s.id === app.selection.focus) || open.scenes.find(isGenerative));
  const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
  return root ? (root.text || "").trim() : undefined;
}
const firstOf = (r) => Object.values(r.rewrites || {})[0];

/** Save one comment on the last rewrite (`last`: the enhancer's latest run, or null). Shared by the Chat tab and the Comment button. */
export function addComment(app, text, last, tail = last) {
  const p = app.project, comment = String(text || "").trim(), now = Array.isArray(p.pref("enhance_chat", [])) ? p.pref("enhance_chat", []) : [];
  if (!comment) return false;
  const ok = last && last.ok, prev = now.length ? firstOf(now[now.length - 1]) : null, here = typedOf(app);
  const fresh = ok && (last.answered ?? last.after) !== prev && app.lastRun.typed !== undefined && app.lastRun.typed === here;      // a rewrite of THIS prompt that nobody has commented on yet
  p.setPref("enhance_chat", [...now, { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, rewrites: fresh ? { whole: last.answered ?? last.after } : {}, comment, ...(tail && tail.prompt_id ? { run: tail.prompt_id } : {}), ...(here === undefined ? {} : { original: here }) }]);
  return true;
}

/** The nearest ancestor that scrolls (the Composer window's body). */
function scrollParent(node) {
  for (let n = node.parentElement; n; n = n.parentElement) { const o = getComputedStyle(n).overflowY; if (o === "auto" || o === "scroll") return n; }
  return null;
}

export function chat(app, own) {
  const p = app.project, page = c.region.stack({ gap: "sm" });
  let last = null, all = [], replying = false, stick = true, alive = true, key = "", draft = "", runs = app.lastRun.n;
  own(() => { alive = false; });
  const rounds = () => { const r = p.pref("enhance_chat", []); return Array.isArray(r) ? r : []; };
  const first = firstOf;
  // The conversation opens at its end (newest message and the message box); "Top ^" goes back to the start.
  const scrollTo = (end) => { const sp = scrollParent(page.node); if (sp) sp.scrollTop = end ? sp.scrollHeight : 0; };
  const later = (f) => (typeof requestAnimationFrame === "function" ? requestAnimationFrame(f) : setTimeout(f, 0));
  function draw(force) {
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
    const noted = (r) => [...(r.remember || []).map((x) => `Noted: ${x}`), ...(r.forget || []).map((x) => `Dropped note ${x}`)].map((text) => c.hint.default({ text }));
    let bubbles;
    if (convo.length) {                      // "Send everything into chat": the whole conversation, each comment under the rewrite it was about
      const asked = new Set(convo.map((r) => r.prompt_id));
      bubbles = [...convo.flatMap((r) => [you(r.asked ?? r.before), model(r.answered ?? r.after), ...noted(r), ...(r.compacted ? [c.hint.default({ text: "The earlier conversation was summarised here to fit the context window." }), ...(r.summary ? [c.collapsible.default({ label: "Summary the model now works from", body: c.text.sm({ text: r.summary }) })] : [])] : []), ...list.filter((x) => x.run === r.prompt_id).map((x) => you(x.comment))]), ...list.filter((x) => !asked.has(x.run)).map((x) => you(x.comment))];
    } else {
      bubbles = [...list.flatMap((r) => [first(r) ? model(first(r)) : null, you(r.comment)]),
        ...(ok ? [c.hint.default({ text: `Last rewrite of: ${String(last.asked ?? last.before ?? "").slice(0, 80)}` }), model(last.answered ?? last.after)] : []),
        ...(!list.length && !ok ? [c.hint.default({ text: "Generate once with the enhancer on, then comment here." })] : [])];
    }
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
    const canChat = inp.to_chat === true && Boolean(inp.encoder_file);
    const say = async () => {
      const message = draft.trim();
      if (!message || replying) return;
      replying = true; draft = ""; draw(true);
      const pick = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "max_length", "compact", "context_tokens", "thinking", "greedy", "encoder_file", "encoder_family"];
      try {
        const r = await app.api.enhancerChat({ conversation: ck, message, notes: String(p.pref("enhance_notes", "") || ""), ...Object.fromEntries(pick.filter((k) => inp[k] !== undefined).map((k) => [k, inp[k]])) });
        if (r && r.compacted) c.toast.warn({ text: "The conversation was getting long: the earlier turns were summarised." });
        if (r && ((r.remember || []).length || (r.forget || []).length)) p.setPref("enhance_notes", applyNotes(p.pref("enhance_notes", ""), r.remember, r.forget));
      } catch (e) { draft = message; c.toast.warn({ text: `No reply: ${(e && e.message) || e}` }); }
      replying = false; stick = true; look(); draw(true);
    };
    page.set([c.label.section({ text: "Chat" }), c.hint.default({ text: "Send gets an answer now. Comment is kept and goes with the next prompt as part of one message. With \"Send everything into chat\" on (Enhance tab), every request and answer shows here. Reset starts over." }),
      c.region.stack({ gap: "sm", children: bubbles }), notesBox, box,
      c.toolbar.default({ items: [c.button.sm({ label: replying ? "Replying…" : "Send", tone: "primary", disabled: !canChat || replying, title: canChat ? "Send a message and get an answer now" : "Needs \"Send everything into chat\" and an Encoder of its own in the Enhance tab", onClick: say }), c.button.sm({ label: "Comment", title: "Kept for the next Generate: it goes with that prompt as part of one message", onClick: comment })], trailing: [c.button.sm({ label: "Refresh", tone: "ghost", title: "Reload the conversation from the enhancer", onClick: () => { stick = true; look().then(() => draw(true)); } }), c.button.sm({ label: "Top ^", tone: "ghost", title: "Back to the start of the conversation", onClick: () => scrollTo(false) }), c.button.sm({ label: "Reset", tone: "ghost", disabled: !list.length && !convo.length, onClick: () => { p.setPref("enhance_chat", []); p.setPref("enhance_chat_epoch", (Number(p.pref("enhance_chat_epoch", 0)) || 0) + 1); draw(true); } })] })]);
  }
  const look = () => app.api.enhancerRuns().then((r) => { all = r.runs || []; last = all.filter((x) => x.kind !== "chat").slice(-1)[0] || null; draw(); if (stick) { stick = false; later(() => scrollTo(true)); } }).catch(() => {});
  own(app.on((what) => { if (what === "open" || app.lastRun.n !== runs) { runs = app.lastRun.n; look(); } else draw(); }));
  if (app.pipeline && app.pipeline.subscribe) own(app.pipeline.subscribe(() => draw()));      // a pipeline that gains or loses the enhancer
  draw(true);
  look();
  return page;
}
