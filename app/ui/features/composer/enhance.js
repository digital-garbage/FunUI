// Enhance: a language model rewrites the finished prompt before it is encoded. These are the inputs of the pipeline's
// enhancer node, edited through the shared pipeline state; off, or on failure, the prompt runs as typed.
import { composer as c } from "../../composer/composer.js";
import { RATING_PREFIX } from "../rating/choices.js";

const NODE = "FunUIEnhancePrompt";
const DIALS = [["temperature", 0.7, 0, 2, 0.05, "Higher gives more varied wording, lower more predictable."], ["top_p", 0.92, 0, 1, 0.01, "Picks only among the likeliest words whose chances add up to this share."],
  ["top_k", 50, 1, 1000, 1, "Picks only among this many likeliest words."], ["min_p", 0.05, 0, 1, 0.01, "Drops words much less likely than the likeliest one; higher drops more."],
  ["repetition_penalty", 1.3, 1, 2, 0.05, "Above 1 discourages repeating words."], ["presence_penalty", 0, 0, 2, 0.05, "Above 0 discourages words already used."]];

export function enhance(app, own) {
  const ps = app.pipeline, page = c.region.stack({ gap: "sm" });
  let defaults = null, last = null, alive = true, engine = { phase: "idle", message: "", downloaded: 0, total: 0 }, poll = 0, encoders = { files: [], families: [] };
  own(() => { alive = false; clearTimeout(poll); });
  const gb = (n) => `${(n / 2 ** 30).toFixed(1)} GB`;
  const engineLine = () => ({ idle: "Not loaded yet. It loads the first time it is used, or press Load now.", loading: `Loading… ${engine.message || ""}`, ready: `Ready: ${engine.message || ""}`,
    downloading: `Downloading: ${gb(engine.downloaded || 0)}${engine.total ? ` of ${gb(engine.total)}` : ""}`, error: `Could not load: ${engine.message || ""}` })[engine.phase] || "";
  const refreshEngine = () => app.api.enhancerEngine().then((r) => { engine = r; if (alive && !app.dom.typingIn(page.node)) draw(); clearTimeout(poll); if (alive && (r.phase === "downloading" || r.phase === "loading")) poll = setTimeout(refreshEngine, 1500); }).catch(() => {});
  const slot = () => (ps.slots() || []).find((s) => s.node === NODE) || null;
  const set = async (name, value) => { const sl = slot(); if (sl) { await ps.save({ inputs: { [sl.id]: { [name]: value } } }); draw(); } };
  const continuityOn = () => { const k = (ps.currentValues && ps.currentValues() || {}).continuity; return Boolean(k) && k.carry !== false; };      // hidden while Continuity is not there or not carrying the frame
  const field = (label, control, hint) => c.field.default({ label, control, hint });
  const num = (label, v, lo, hi, step, name, hint, disabled) => field(label, c.number.md({ label, value: v, min: lo, max: hi, step, disabled, onChange: (x) => set(name, x) }), hint);

  function readout() {
    if (!last) return [c.label.section({ text: "Last run" }), c.hint.default({ text: "Nothing yet. The rewrite of a run shows here." })];
    const part = (title, body) => (body ? c.collapsible.default({ label: title, body: c.code.block({ text: body, label: title }) }) : null);
    return [c.label.section({ text: "Last run" }), c.hint.default({ text: last.status || "" }), part("Before", last.before),
      c.field.default({ label: "After", control: c.text.sm({ text: last.after || "" }) }), part("Sent to the model", last.sent)].filter(Boolean);
  }
  // Everything the model is told, in the order a request meets it. `dflt`: what an empty field means.
  const PROMPTS = [
    ["instructions", "Rewrite instructions", "What the model does with each prompt. The Style above decides the built-in text.", 6, () => (defaults ? (defaults.styles || {})[styleNow()] || defaults.instructions : "")],
    ["marks_prompt", "Markup rule", "Keeps (word:1.5) and [a|b] as written.", 4],
    ["reference_intro", "Reference intro", "Opens the lorebook and shortcut entries sent along with a prompt.", 3, () => (defaults ? defaults.reference_intro : "")],
    ["conversation_prompt", "Conversation, plain encoders", "Sent with the earlier turns to an encoder that takes no chat turns.", 5, null, "chat"],
    ["turns_prompt", "Conversation, Gemma 4", "Sent to a model that takes real chat turns.", 5, null, "chat"],
    ["revision_prompt", "Comments", "How comments are applied when there is no conversation.", 5, null, "nochat"],
    ["rating_prefix", "Rating line opening", "The words in front of the rating when Pass the rating is on.", 2, () => RATING_PREFIX, "rate"],
    ["chat_prompt", "Chat answers", "Send in the Chat tab: how the model answers you.", 5, null, "talk"],
    ["notes_prompt", "Notes rule", "How the model adds and drops Notes in a chat answer.", 5, null, "talk"],
    ["compact_prompt", "Conversation notes", "The model that shortens a long conversation.", 6, null, "sum"]];
  const styleNow = () => { const x = (slot() && slot().inputs) || {}; return x.style || "default"; };
  const choices = (name) => (name === "encoder_file" ? encoders.files : encoders.families);
  const pick = (name, label, first, hint, dflt = "", disabled = false) => {
    const cur = (slot().inputs || {})[name] || dflt, list = choices(name);
    const opts = [...(first ? [{ value: "", label: first }] : []), ...[...list, ...(cur && !list.includes(cur) ? [cur] : [])].map((v) => ({ value: v, label: String(v) }))];
    return field(label, c.select.md({ label, value: cur, options: opts, disabled, onChange: (v) => set(name, v) }), hint);
  };
  function draw() {
    if (!alive) return;
    const sl = slot();
    if (!sl) return page.set([c.emptyState.default({ icon: "✎", title: "No prompt enhancer here", hint: "Pick a preset that has one, or add the “FunUI Enhance Prompt” node in Settings ▸ Models & Pipeline." })]);
    const v = sl.inputs || {}, val = (n, d) => (v[n] === undefined ? d : v[n]);
    const on = val("enabled", false), greedy = val("greedy", false);
    const styles = defaults && defaults.styles ? Object.keys(defaults.styles) : ["default", "dynamic"], style = styles.includes(val("style", "default")) ? val("style", "default") : "default";
    // A setting that hangs on another is greyed out, not hidden, while that one is off: `dep` names what it hangs on.
    const engineSet = Boolean(String(val("engine_model", "")).trim()), chat = val("to_chat", false), sum = val("compact", true), rate = val("pass_rating", false), encoder = Boolean(val("encoder_file", "")), pic = val("use_image", true), cont = continuityOn();
    const live = { chat, nochat: !chat, rate: chat && rate, talk: chat && (encoder || engineSet), sum: chat && sum };
    const off = (...needs) => !on || needs.some((n) => !n);
    const toggle = (label, hint, name, dflt, disabled) => c.toggle.default({ label, hint, checked: val(name, dflt), disabled, onChange: (x) => set(name, x) });
    page.set([
      c.label.section({ text: "Prompt enhancer" }),
      c.hint.default({ text: "A language model rewrites the finished prompt. Off or failed, your prompt is used as typed. (word:1.5) and [a|b] markup is kept as written. Settings that depend on another are greyed out while that one is off." }),
      toggle("Enhance prompt", "", "enabled", false, false),
      c.label.section({ text: "Model" }),
      field("Engine model", c.input.sm({ label: "Engine model", value: val("engine_model", ""), placeholder: "owner/name on Hugging Face, or a folder", disabled: off(), onCommit: (x) => { set("engine_model", x.trim()); refreshEngine(); } }),
        "Runs the model on FunUI's own engine, which can use any model that transformers supports and shows the chat as it is written. The first use downloads it. Empty uses the encoder below."),
      c.hint.default({ text: engineSet ? engineLine() : "No engine model set: the encoder below writes the rewrites." }),
      c.toolbar.default({ items: [c.button.sm({ label: "Load now", disabled: off(engineSet) || engine.phase === "loading" || engine.phase === "downloading", onClick: () => app.api.enhancerEngineLoad(val("engine_model", "")).then(refreshEngine).catch((e) => c.toast.warn({ text: `Could not start loading: ${e.message}` })) }),
        c.button.sm({ label: "Unload", tone: "ghost", disabled: off(engineSet) || engine.phase !== "ready", onClick: () => app.api.enhancerEngineUnload().then(refreshEngine) })] }),
      pick("encoder_file", "Encoder", "The model's own encoder", "The text encoder that writes the rewrite when no engine model is set. Models like H3's own encoder cannot write prose: pick a chat model such as Gemma 4 here.", "", off(!engineSet)),
      pick("encoder_family", "Encoder family", "", "Which family the encoder file is for.", "stable_diffusion", off(encoder, !engineSet)),
      field("Rewrites", c.select.md({ label: "Rewrites", value: val("scope", "whole"), disabled: off(), options: [{ value: "whole", label: "The whole prompt" }, { value: "own", label: "Only my own text" }], onChange: (x) => set("scope", x) }), "Only my own text keeps the global prefix and postfix exactly as written and rewrites what sits between them."),
      c.label.section({ text: "Picture" }),
      toggle("Use the picture", "Sends the scene's picture to the model along with the prompt.", "use_image", true, off()),
      toggle("Pass the last frame of previous scene as enhancer input", "Shows the model the previous scene's last frame instead of the first reference. The first scene uses the first reference. Needs Continuity on.", "last_frame", false, off(pic, cont)),
      c.label.section({ text: "Conversation" }),
      toggle("Send everything into chat", "Keeps one running conversation with the model: each request is sent together with the earlier requests and its answers, instead of as a one-time request. Reset in the Chat tab starts it over.", "to_chat", false, off()),
      toggle("Summarise long conversations", "When a request would nearly fill the model's context window, the conversation so far is summarised and the summary is sent instead of every turn.", "compact", true, off(chat)),
      num("Context window (tokens)", val("context_tokens", 0), 0, 1048576, 256, "context_tokens", "Summarising starts at 80% of this. 0 uses 32768, or the model's own size if smaller.", off(chat, sum)),
      toggle("Pass the rating", "Part of the conversation: adds the rating of the last rated scene before this one to the end of the request.", "pass_rating", false, off(chat)),
      toggle("Comment button", "Shows a Comment button next to a scene's rating. A comment goes to the model with the next Generate, the same as in the Chat tab.", "comment_button", false, off()),
      c.label.section({ text: "Writing" }),
      num("Max tokens", val("max_length", 400), 32, 4096, 8, "max_length", "The longest the rewrite may be.", off()),
      toggle("Thinking", "Let a thinking model reason first. Its reasoning eats the token budget.", "thinking", false, off()),
      toggle("Greedy (always the likeliest word)", "The same rewrite every time; the dials below are not used.", "greedy", false, off()),
      ...DIALS.map(([n, d, lo, hi, st]) => num(n, val(n, d), lo, hi, st, n, undefined, off(!greedy))),
      num("Seed", val("seed", 0), 0, 2 ** 53, 1, "seed", undefined, off()),
      c.hint.default({ text: "Seed 0 = a fresh rewrite every Generate; any other number repeats the same rewrite." }),
      c.label.section({ text: "System prompts" }),
      c.collapsible.default({ label: "Everything the model is told", body: c.region.stack({ gap: "xs", children: [
        c.hint.default({ text: "In the order a request meets it. Leave a field empty to use the built-in text shown in it. Any of these can use the macros at the end: each is replaced by what the conversation holds when the model is asked. A field that does not apply to the current switches is greyed out." }),
        field("Style", c.select.md({ label: "Style", value: style, disabled: off(), options: styles.map((n) => ({ value: n, label: n === "dynamic" ? "Dynamic (more motion)" : "Default" })), onChange: (x) => set("style", x) }), "Which built-in rewrite instructions an empty Rewrite instructions field means. Dynamic asks for a concrete action beat, a change over time and secondary motion in every shot."),
        ...PROMPTS.map(([n, label, hint, rows, dflt, dep]) => field(label, c.textarea.md({ label, rows, disabled: off(dep ? live[dep] : true), value: val(n, ""), placeholder: dflt ? dflt() : defaults && defaults.prompts ? defaults.prompts[n] || "" : "(built-in text)", onCommit: (x) => set(n, x) }), hint)),
        c.collapsible.default({ label: "Macros", body: c.region.stack({ gap: "xs", children: [
          ...Object.entries((defaults && defaults.macros) || {}).map(([n, d]) => c.hint.default({ text: `[[${n}]]: ${d}` })),
          c.hint.default({ text: "Nothing to show yet gives \"(nothing yet: none was given so far)\". A name that is not in this list stays as written. Turns that were shortened are only in [[CONVERSATION_SUMMARY]]." })] }) })] }) }),
      c.label.section({ text: "Reference" }),
      field("Lorebook / shortcut files", c.textarea.md({ label: "Lorebook files", rows: 2, disabled: off(), value: val("lorebooks", ""), placeholder: "one path per line", onCommit: (x) => set("lorebooks", x) }), "SillyTavern lorebooks or FunUI shortcut files, on the machine running ComfyUI."),
      ...readout(),
    ]);
  }
  draw();
  refreshEngine();
  ps.ensureLoaded().then(draw);
  app.api.enhancerEncoders().then((r) => { encoders = { files: r.files || [], families: r.families || [] }; draw(); }).catch(() => {});
  // Another project's pipeline, or an edit made elsewhere: redraw, but never under a field being typed in.
  const typing = () => app.dom.typingIn(page.node);
  own(ps.subscribe(() => { if (!typing()) draw(); }));
  app.api.enhancerDefaults().then((d) => { defaults = d; draw(); }).catch(() => {});
  const fetchRuns = () => app.api.enhancerRuns().then((r) => { last = (r.runs || []).filter((x) => x.kind !== "chat").slice(-1)[0] || null; if (!typing()) draw(); }).catch(() => {});
  fetchRuns();
  // a run that ends while this tab is open shows its prompt here without reopening it
  let phase = null;
  if (app.generate && app.generate.subscribe) own(app.generate.subscribe((st) => {
    const was = phase; phase = st && st.phase;
    if (was && was !== phase && (phase === "done" || phase === "failed")) fetchRuns();
  }));
  return page;
}
