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
  let defaults = null, last = null, alive = true, engine = { phase: "idle", message: "", downloaded: 0, total: 0 }, poll = 0, encoders = { files: [], families: [] }, repoFiles = { repo: "", files: [], why: "" }, downloads = [], writing = null, tick = 0, watchUntil = 0;
  own(() => { alive = false; clearTimeout(poll); clearTimeout(tick); });
  const gb = (n) => `${(n / 2 ** 30).toFixed(1)} GB`;
  const engineLine = () => ({ idle: "Not loaded yet. It loads the first time it is used, or press Load now.", loading: `Loading… ${engine.message || ""}`, ready: engine.busy && engine.busy.stage ? `${engine.busy.stage}${engine.busy.tokens ? `: ${engine.busy.tokens} tokens` : engine.busy.prompt ? ` (${engine.busy.prompt} tokens)` : ""}, ${engine.busy.seconds}s` : `Ready: ${engine.message || ""}`,
    downloading: `Downloading${engine.label ? ` ${engine.label}` : ""}: ${gb(engine.downloaded || 0)}${engine.total ? ` of ${gb(engine.total)}` : ""}`, error: `Could not load: ${engine.message || ""}` })[engine.phase] || "";
  const refreshEngine = () => app.api.enhancerEngine().then((r) => {
    const changed = JSON.stringify(r) !== JSON.stringify(engine);
    const was = engine.phase;
    engine = r;
    if ((was === "downloading" && r.phase !== "downloading") || (changed && r.phase !== "downloading" && /downloaded|download failed/.test(r.message || ""))) refreshDownloads();
    if (changed && alive && !app.dom.typingIn(page.node)) draw();
    clearTimeout(poll);
    if (alive) { poll = setTimeout(refreshEngine, r.phase === "downloading" || r.phase === "loading" || Date.now() < watchUntil || (r.busy && r.busy.stage) ? 1000 : 5000); if (poll && poll.unref) poll.unref(); }     // keeps watching: a model that works or loads in the background shows it here
  }).catch(() => {});
  // The weight files of the typed Hugging Face repo, so one of them can be picked (an fp8 file without the bf16 ones).
  const repoOf = (v) => { const m = /^([\w.-]+\/[\w.-]+)(?:::([^@]+))?(?:@(.+))?$/.exec(String(v || "").trim()); return m ? { repo: m[1], file: m[2] || "", base: m[3] || "" } : null; };
  const listRepo = (v) => { const r = repoOf(v); if (!r || r.repo === repoFiles.repo) return; repoFiles = { repo: r.repo, files: [], why: "Looking…" }; Promise.resolve().then(() => app.api.enhancerEngineFiles(r.repo)).then((x) => { if (repoFiles.repo === r.repo) { repoFiles = { repo: r.repo, files: x.files || [], why: x.why || "" }; if (alive && !app.dom.typingIn(page.node)) draw(); } }).catch(() => {}); };
  const refreshDownloads = () => Promise.resolve().then(() => app.api.enhancerEngineModels()).then((r) => { downloads = r.models || []; if (alive && !app.dom.typingIn(page.node)) draw(); }).catch(() => {});
  const forget = async (m) => {
    if (!(await c.modal.dialogue({ title: "Delete downloaded model", message: `Delete ${m.name.replace("__", "/")} (${gb(m.size)}) from disk? It downloads again the next time it is used.${m.loaded ? " It is loaded now and will be unloaded." : ""}`, tone: "danger", confirmLabel: "Delete" }).result)) return;
    try { await app.api.enhancerEngineDelete(m.name); } catch (e) { c.toast.warn({ text: `Could not delete: ${e.message}` }); }
    refreshEngine(); refreshDownloads();
  };
  const setMany = async (inputs) => { const sl = slot(); if (sl) { await ps.save({ inputs: { [sl.id]: inputs } }); draw(); } };
  const slot = () => (ps.slots() || []).find((s) => s.node === NODE) || null;
  const set = async (name, value) => { const sl = slot(); if (sl) { await ps.save({ inputs: { [sl.id]: { [name]: value } } }); draw(); } };
  const continuityOn = () => { const k = (ps.currentValues && ps.currentValues() || {}).continuity; return Boolean(k) && k.carry !== false; };      // hidden while Continuity is not there or not carrying the frame
  const field = (label, control, hint) => c.field.default({ label, control, hint });
  const num = (label, v, lo, hi, step, name, hint, disabled) => field(label, c.number.md({ label, value: v, min: lo, max: hi, step, disabled, onChange: (x) => set(name, x) }), hint);

  function readout() {
    const now = writing && !writing.done ? [c.label.section({ text: "Writing now" }), c.hint.default({ text: writing.status || `Writing: ${writing.label}` }),
      ...(writing.thinking ? [c.collapsible.default({ label: "Thinking", open: true, body: c.code.block({ text: writing.thinking, label: "Thinking" }) })] : []), ...(writing.answer ? [c.code.block({ text: writing.answer, label: "Rewrite so far" })] : [])] : [];
    if (!last) return [...now, c.label.section({ text: "Last run" }), c.hint.default({ text: "Nothing yet. The rewrite of a run shows here." })];
    const part = (title, body) => (body ? c.collapsible.default({ label: title, body: c.code.block({ text: body, label: title }) }) : null);
    return [...now, c.label.section({ text: "Last run" }), c.hint.default({ text: last.status || "" }), part("Before", last.before),
      c.field.default({ label: "After", control: c.text.sm({ text: last.after || "" }) }), part("Sent to the model", last.sent)].filter(Boolean);
  }
  // Everything the model is told, in the order a request meets it. `dflt`: what an empty field means.
  const PROMPTS = [
    ["instructions", "Rewrite instructions", "What the model does with each prompt. The Style above decides the built-in text.", 6, () => (defaults ? (defaults.styles || {})[styleNow()] || defaults.instructions : "")],
    ["marks_prompt", "Markup rule", "Keeps (word:1.5) and [a|b] as written.", 4],
    ["fixed_prompt", "Fixed text around it", "Tells the model what the global prefix and postfix say, so it writes for the whole prompt.", 4, null, "own"],
    ["reference_intro", "Reference intro", "Opens the lorebook and shortcut entries sent along with a prompt.", 3, () => (defaults ? defaults.reference_intro : "")],
    ["conversation_prompt", "Conversation, plain encoders", "Sent with the earlier turns to an encoder that takes no chat turns.", 5, null, "chat"],
    ["turns_prompt", "Conversation, Gemma 4", "Sent to a model that takes real chat turns.", 5, null, "chat"],
    ["revision_prompt", "Comments", "How comments are applied when there is no conversation.", 5, null, "nochat"],
    ["rating_prefix", "Rating line opening", "The words in front of the rating when Pass the rating is on.", 2, () => RATING_PREFIX, "rate"],
    ["chat_prompt", "Chat answers", "Send in the Chat tab: how the model answers you.", 5, null, "talk"],
    ["edit_prompt", "Editing its own rules", "Tells the chat model it can read and propose changes to its own rules; [[RULES_SHEET]] is the list.", 5, null, "edit"],
    ["refine_prompt", "Self-check", "What the model is told in each self-check pass over its own rewrite.", 5, null, "refine"],
    ["notes_prompt", "Notes rule", "How the model adds and drops Notes in a chat answer.", 5, null, "talk"],
    ["compact_prompt", "Conversation notes", "The model that shortens a long conversation.", 6, null, "sum"]];
  const styleNow = () => { const x = (slot() && slot().inputs) || {}; return x.style || "default"; };
  function draw() {
    if (!alive) return;
    const sl = slot();
    if (!sl) return page.set([c.emptyState.default({ icon: "✎", title: "No prompt enhancer here", hint: "Pick a preset that has one, or add the “FunUI Enhance Prompt” node in Settings ▸ Models & Pipeline." })]);
    const v = sl.inputs || {}, val = (n, d) => (v[n] === undefined ? d : v[n]);
    const on = val("enabled", false), greedy = val("greedy", false);
    const styles = defaults && defaults.styles ? Object.keys(defaults.styles) : ["default", "dynamic"], style = styles.includes(val("style", "default")) ? val("style", "default") : "default";
    // A setting that hangs on another is greyed out, not hidden, while that one is off: `dep` names what it hangs on.
    const engineSet = !val("use_comfy", false) && Boolean(String(val("engine_model", "")).trim() || val("encoder_file", "")), chat = val("to_chat", false), sum = val("compact", true), rate = val("pass_rating", false), encoder = Boolean(val("encoder_file", "")), pic = val("use_image", true), cont = continuityOn();
    const live = { own: val("scope", "whole") === "own", chat, nochat: !chat, rate: chat && rate, talk: chat && engineSet, edit: chat && engineSet && val("self_edit", false), refine: val("refine_passes", 0) > 0, sum: chat && sum };
    const off = (...needs) => !on || needs.some((n) => !n);
    const toggle = (label, hint, name, dflt, disabled) => c.toggle.default({ label, hint, checked: val(name, dflt), disabled, onChange: (x) => set(name, x) });
    page.set([
      c.label.section({ text: "Prompt enhancer" }),
      c.hint.default({ text: "A language model rewrites the finished prompt. Off or failed, your prompt is used as typed. Markup like (word:1.5) is kept. Dependent settings are greyed out while their switch is off." }),
      toggle("Enhance prompt", "", "enabled", false, false),
      c.label.section({ text: "Model" }),
      field("Engine model", c.input.sm({ label: "Engine model", value: val("engine_model", ""), placeholder: "owner/name on Hugging Face, or a folder", disabled: off(), onCommit: (x) => { set("engine_model", x.trim()); refreshEngine(); } }),
        "Runs the model on FunUI's own engine, which can use any model that transformers supports and shows the chat as it is written. The first use downloads it. For the fastest writing pick a GGUF repo (for example a Q4_K_M or Q8_0 file): the engine runs those with llama.cpp, which is installed on first use. fp8 files save memory, not time."),
      field("From ComfyUI's list", c.select.md({ label: "From ComfyUI's list", value: String(val("engine_model", "")).startsWith("comfy:") ? String(val("engine_model", "")).slice(6).split("@")[0] : (String(val("engine_model", "")).trim() ? "" : String(val("encoder_file", ""))), disabled: off(), options: [{ value: "", label: "None" }, ...encoders.files.map((f) => ({ value: f, label: f }))], onChange: (x) => { setMany({ engine_model: x ? `comfy:${x}` : "", encoder_file: "" }).then(refreshEngine); } }),
        "Uses a text encoder file from ComfyUI's models list as the engine model. Its config and tokenizer are downloaded from Hugging Face once; add @owner/name to the model above if the right one is not found."),
      ...(() => { const r = repoOf(val("engine_model", "")); if (!r) return []; listRepo(val("engine_model", "")); const sel = repoFiles.repo === r.repo ? repoFiles.files : [];
        const picked = r.file.split("|").filter(Boolean);
        const choose = (name, on) => { const next = sel.map((f) => f.name).filter((n) => (n === name ? on : picked.includes(n))); set("engine_model", next.length ? `${r.repo}::${next.join("|")}${r.base ? `@${r.base}` : ""}` : r.repo).then(refreshEngine); };
        return sel.length > 1 || r.file ? [c.label.section({ text: "Files in the repo" }), c.hint.default({ text: "Nothing checked downloads the whole repo. Check only the files you need (one fp8 file, or every part of a split .gguf model), then press Load now." }),
          ...sel.map((f) => c.toggle.default({ label: `${f.name} (${gb(f.size)})${downloads.some((m) => String(m.source || "").startsWith(`${r.repo}::${f.name}`)) ? ", downloaded" : ""}`, checked: picked.includes(f.name), disabled: off(), onChange: (x) => choose(f.name, x) }))] : []; })(),
      toggle("Use ComfyUI's text generation instead", "Writes with ComfyUI's own text generation (the pipeline's text encoder, or the file picked below) and leaves FunUI's engine off. No streaming, no model editing its own rules.", "use_comfy", false, off()),
      c.hint.default({ text: engineSet ? engineLine() : "No model chosen: the pipeline's own text encoder writes the rewrites, and it cannot chat." }),
      c.toolbar.default({ items: [c.button.sm({ label: "Download", tone: "ghost", title: "Download the model's files only (the checked ones); nothing is loaded into memory and the loaded model stays", disabled: off(engineSet) || engine.phase === "loading" || engine.phase === "downloading", onClick: () => app.api.enhancerEngineDownload(val("engine_model", "")).then((r) => { watchUntil = Date.now() + 15000; if (r && r.started === false) c.toast.warn({ text: "Not started: a download or a load is still running. Try again when it ends." }); refreshEngine(); refreshDownloads(); }).catch((e) => c.toast.warn({ text: `Could not start downloading: ${e.message}` })) }), c.button.sm({ label: "Load now", disabled: off(engineSet) || engine.phase === "loading" || engine.phase === "downloading", onClick: () => app.api.enhancerEngineLoad(val("engine_model", ""), val("precision", "auto")).then(refreshEngine).catch((e) => c.toast.warn({ text: `Could not start loading: ${e.message}` })) }),
        c.button.sm({ label: "Unload", tone: "ghost", disabled: off(engineSet) || engine.phase !== "ready", onClick: () => app.api.enhancerEngineUnload().then(refreshEngine) })] }),
      ...(downloads.length ? [c.collapsible.default({ label: `Downloaded models (${downloads.length})`, body: c.region.stack({ gap: "xs", children: downloads.map((m) => c.toolbar.default({ items: [c.text.sm({ text: `${m.name.replace("__", "/")}: ${gb(m.size)}${m.loaded ? " (loaded)" : ""}` })], trailing: [c.button.sm({ label: "Use", tone: "ghost", title: "Use this downloaded model as the engine model", disabled: engine.phase === "loading" || engine.phase === "downloading", onClick: () => { set("engine_model", m.source || m.name.replace("__", "/")); app.api.enhancerEngineLoad(m.source || m.name.replace("__", "/"), val("precision", "auto")).then(refreshEngine).catch(() => {}); } }), c.button.sm({ label: "Delete", tone: "ghost", title: "Remove this model's downloaded files from disk", onClick: () => forget(m) })] })) }) })] : []),
      field("Rewrites", c.select.md({ label: "Rewrites", value: val("scope", "whole"), disabled: off(), options: [{ value: "whole", label: "The whole prompt" }, { value: "own", label: "Only my own text" }], onChange: (x) => set("scope", x) }), "Only my own text keeps the global prefix and postfix exactly as written and rewrites what sits between them."),
      c.label.section({ text: "Picture" }),
      toggle("Use the picture", "Sends the scene's picture to the model along with the prompt.", "use_image", true, off()),
      toggle("Pass the last frame of previous scene as enhancer input", "Shows the model the previous scene's last frame instead of the first reference. The first scene uses the first reference. Needs Continuity on.", "last_frame", false, off(pic, cont)),
      c.label.section({ text: "Conversation" }),
      toggle("Send everything into chat", "Keeps one running conversation with the model: each request is sent together with the earlier requests and its answers, instead of as a one-time request. Reset in the Chat tab starts it over.", "to_chat", false, off()),
      toggle("Summarise long conversations", "When a request would nearly fill the model's context window, the conversation so far is summarised and the summary is sent instead of every turn.", "compact", true, off(chat)),
      num("Context window (tokens)", val("context_tokens", 0), 0, 1048576, 256, "context_tokens", "Summarising starts at 80% of this. 0 uses 32768, or the model's own size if smaller.", off(chat, sum)),
      toggle("Pass the rating", "Part of the conversation: adds the rating of the last rated scene before this one to the end of the request.", "pass_rating", false, off(chat)),
      toggle("Let the model edit its own rules", "The model can read its system prompts and propose changes to them in the Chat tab. Each change is shown before/after and applies only if you allow it. Needs a model with tool calls (Gemma 4, Qwen 3 and most newer chat models).", "self_edit", false, off(chat, engineSet)),
      toggle("Comment button", "Shows a Comment button next to a scene's rating. A comment goes to the model with the next Generate, the same as in the Chat tab.", "comment_button", false, off()),
      c.label.section({ text: "Speed" }),
      field("Precision", c.select.md({ label: "Precision", value: val("precision", "auto"), disabled: off(engineSet), options: [{ value: "auto", label: "Automatic" }, { value: "bf16", label: "bf16 (fastest, double memory)" }, { value: "fp8", label: "fp8 (smallest)" }] , onChange: (x) => set("precision", x) }), "How a single fp8 model file is run. Automatic widens it to bf16 once when the graphics card has room, which makes every word faster; on the processor it stays fp8. Applies the next time the model loads."),
      toggle("Reuse the unchanged start of a request", "The start of a request that did not change since the last one (the rules, the conversation) is not read again, only the new part is.", "prefix_cache", true, off(engineSet)),
      num("Guess ahead (words)", val("lookup_tokens", 0), 0, 10, 1, "lookup_tokens", "The model may guess up to this many words from the prompt itself and check them in one step. Only helps when a rewrite mostly repeats the prompt, and costs time otherwise: off by default (0).", off(engineSet)),
      c.label.section({ text: "Writing" }),
      num("Max tokens", val("max_length", 0), 0, 4096, 8, "max_length", "The longest the rewrite may be. 0 means no limit: the model stops by itself, or when the context window is full.", off()),
      num("Self-check passes", val("refine_passes", 0), 0, 3, 1, "refine_passes", "How many times the model checks its own rewrite against the rules and improves it. Each pass is another model call, so a rewrite takes longer.", off()),
      toggle("Thinking", "Let a thinking model reason first. Its reasoning eats the token budget.", "thinking", false, off()),
      toggle("Greedy (always the likeliest word)", "The same rewrite every time; the dials below are not used.", "greedy", false, off()),
      ...DIALS.map(([n, d, lo, hi, st]) => num(n, val(n, d), lo, hi, st, n, undefined, off(!greedy))),
      num("Seed", val("seed", 0), 0, 2 ** 53, 1, "seed", undefined, off()),
      c.hint.default({ text: "Seed 0 = a fresh rewrite every Generate; any other number repeats the same rewrite." }),
      c.label.section({ text: "System prompts" }),
      c.collapsible.default({ label: "Everything the model is told", body: c.region.stack({ gap: "xs", children: [
        c.hint.default({ text: "In the order a request meets it. An empty field uses the built-in text shown in it. Any field can use the macros listed below. Fields that do not apply are greyed out." }),
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
  refreshDownloads();
  const watch = () => Promise.resolve().then(() => app.api.enhancerLive()).then((r) => {
    const now = r && r.id && !r.done ? r : null;
    if (JSON.stringify(now) !== JSON.stringify(writing)) { const ended = writing && !now; writing = now; if (alive && !app.dom.typingIn(page.node)) { if (ended) app.api.enhancerRuns().then((x) => { last = (x.runs || []).slice(-1)[0] || last; draw(); }).catch(() => draw()); else draw(); } }
  }).catch(() => {}).finally(() => { if (alive) { tick = setTimeout(watch, writing ? 600 : 2500); if (tick && tick.unref) tick.unref(); } });
  watch();
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
