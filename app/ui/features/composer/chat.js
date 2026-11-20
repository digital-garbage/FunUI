// Chat: comment on the enhancer's last rewrite. The next Generate rewrites again with every comment still applying; Reset starts over.
// The comments are the project's own (editor_settings.enhance_chat); a generate hook hands them to the enhancer node for the prompt they were about.
import { composer as c } from "../../composer/composer.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";

const NODE = "FunUIEnhancePrompt";
const COMPACT_AT = 0.8;        // the share of the context window at which the conversation is summarised (the node uses the same)

/** Generate hook: the comments about THIS typed prompt go to the enhancer slot; the others are said to be left out. */
export function chatHook({ project, scene, slots }) {
  const chat = (project.editor_settings || {}).enhance_chat, slot = (slots || []).find((s) => s.node === NODE);
  if (!slot || !Array.isArray(chat)) return null;
  // In a conversation every comment belongs to it, whatever prompt it was made on; otherwise only the ones about this prompt apply.
  const typed = (scene.text || "").trim(), mine = chat.filter((r) => r && ((slot.inputs || {}).to_chat === true || r.original === undefined || r.original === typed)), left = chat.length - mine.length;
  return { inputs: mine.length ? { [slot.id]: { chat: JSON.stringify(mine) } } : {}, notes: left ? [`${left} Chat comment(s) are about a different prompt and were left out of this run.`] : [] };
}

/** The prompt a comment is about: the picked scene's own text, as typed. */
function typedOf(app) {
  const open = app.project.project, sc = open && (open.scenes.find((s) => s.id === app.selection.focus) || open.scenes.find(isGenerative));
  const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
  return root ? (root.text || "").trim() : undefined;
}
const firstOf = (r) => Object.values(r.rewrites || {})[0];

/** Save one comment on the last rewrite (`last`: the enhancer's latest run, or null). Shared by the Chat tab and the Comment button. */
export function addComment(app, text, last) {
  const p = app.project, comment = String(text || "").trim(), now = Array.isArray(p.pref("enhance_chat", [])) ? p.pref("enhance_chat", []) : [];
  if (!comment) return false;
  const ok = last && last.ok, prev = now.length ? firstOf(now[now.length - 1]) : null, here = typedOf(app);
  const fresh = ok && (last.answered ?? last.after) !== prev && app.lastRun.typed !== undefined && app.lastRun.typed === here;      // a rewrite of THIS prompt that nobody has commented on yet
  p.setPref("enhance_chat", [...now, { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, rewrites: fresh ? { whole: last.answered ?? last.after } : {}, comment, ...(last && last.prompt_id ? { run: last.prompt_id } : {}), ...(here === undefined ? {} : { original: here }) }]);
  return true;
}

export function chat(app, own) {
  const p = app.project, page = c.region.stack({ gap: "sm" });
  let last = null, all = [], alive = true, key = "", draft = "", runs = app.lastRun.n;
  own(() => { alive = false; });
  const rounds = () => { const r = p.pref("enhance_chat", []); return Array.isArray(r) ? r : []; };
  const first = firstOf;
  function draw(force) {
    if (!alive) return;
    const here = ((app.pipeline && app.pipeline.slots()) || []).some((s) => s.node === NODE);
    const list = rounds(), ok = last && last.ok, next = JSON.stringify([p.project && p.project.id, list, last && (last.answered ?? last.after), here, all.length, p.pref("enhance_chat_epoch", 0)]);
    if (!force && (next === key || app.dom.typingIn(page.node, "TEXTAREA"))) return;       // not under the caret
    key = next;
    if (!p.project) return page.set([c.hint.default({ text: "Open a project first." })]);
    // Comments would reach nothing: said here, the way Enhance says it (they stay, for a pipeline that has one).
    if (!here) return page.set([c.emptyState.default({ icon: "✎", title: "No prompt enhancer here", hint: "Chat comments go to the enhancer. Pick a preset that has one, or add the “FunUI Enhance Prompt” node in Settings ▸ Models & Pipeline." })]);
    const you = (text) => c.bubble.you({ text, who: "You" }), model = (text) => c.bubble.model({ text, who: "Model" });
    const ck = `${p.project.id}:${Number(p.pref("enhance_chat_epoch", 0)) || 0}`, convo = all.filter((r) => r.conversation === ck && r.ok);
    let bubbles;
    if (convo.length) {                      // "Send everything into chat": the whole conversation, each comment under the rewrite it was about
      const asked = new Set(convo.map((r) => r.prompt_id));
      bubbles = [...convo.flatMap((r) => [you(r.asked ?? r.before), model(r.answered ?? r.after), ...list.filter((x) => x.run === r.prompt_id).map((x) => you(x.comment))]), ...list.filter((x) => !asked.has(x.run)).map((x) => you(x.comment))];
    } else {
      bubbles = [...list.flatMap((r) => [first(r) ? model(first(r)) : null, you(r.comment)]),
        ...(ok ? [c.hint.default({ text: `Last rewrite of: ${String(last.asked ?? last.before ?? "").slice(0, 80)}` }), model(last.answered ?? last.after)] : []),
        ...(!list.length && !ok ? [c.hint.default({ text: "Generate once with the enhancer on, then comment here." })] : [])];
    }
    bubbles = bubbles.filter(Boolean);
    const cx = convo.length && convo[convo.length - 1].context;
    if (cx) bubbles.push(c.hint.default({ text: `Context: about ${cx.used.toLocaleString("en-US")} of ${cx.window.toLocaleString("en-US")} tokens (${Math.round(100 * cx.used / cx.window)}%). ${cx.compact ? `Summarised at ${Math.round(100 * COMPACT_AT)}%.` : "Summarising is off."}` }));      // as of the last request
    const box = c.textarea.md({ label: "Comment", rows: 2, value: draft, placeholder: "e.g. warmer light, no rain", onInput: (v) => { draft = v; } });
    const send = () => { if (addComment(app, draft, last)) { draft = ""; draw(true); } };
    page.set([c.label.section({ text: "Chat" }), c.hint.default({ text: "Comment on the last rewrite. The next Generate rewrites again with every comment still applying. With \"Send everything into chat\" on (Enhance tab), every request and answer shows here. Reset starts over." }),
      c.region.stack({ gap: "sm", children: bubbles }), box,
      c.toolbar.default({ items: [c.button.sm({ label: "Send", tone: "primary", onClick: send })], trailing: [c.button.sm({ label: "Reset", tone: "ghost", disabled: !list.length && !convo.length, onClick: () => { p.setPref("enhance_chat", []); p.setPref("enhance_chat_epoch", (Number(p.pref("enhance_chat_epoch", 0)) || 0) + 1); draw(true); } })] })]);
  }
  const look = () => app.api.enhancerRuns().then((r) => { all = r.runs || []; last = all.slice(-1)[0] || null; draw(); }).catch(() => {});
  own(app.on((what) => { if (what === "open" || app.lastRun.n !== runs) { runs = app.lastRun.n; look(); } else draw(); }));
  if (app.pipeline && app.pipeline.subscribe) own(app.pipeline.subscribe(() => draw()));      // a pipeline that gains or loses the enhancer
  draw(true);
  look();
  return page;
}
