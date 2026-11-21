// ◆ Composer: the floating "prompt craft" window. Opens from the timeline header; its tabs are the ones v4 had.
import { composer as c } from "../../composer/composer.js";
import { applyStory, clash, joinStory, shownStory } from "./story.js";
import { shortcuts } from "./shortcuts.js";
import { cuts } from "./cuts.js";
import { enhance } from "./enhance.js";
import { build } from "./build.js";
import { chat, chatHook } from "./chat.js";
import { openHelper } from "./helper.js";
import { files } from "./files.js";
import { pickShortcut, templatesBar, variablesPanel } from "./storytools.js";

const TABS = [{ value: "story", label: "Story" }, { value: "shortcuts", label: "Shortcuts" }, { value: "build", label: "Build" }, { value: "cuts", label: "Cuts" }, { value: "enhance", label: "Enhance" }, { value: "chat", label: "Chat" }, { value: "files", label: "Files" }];

// The box follows the scenes; typing in it rewrites them (after a short pause, or when focus leaves).
export const story = (app, own) => {
  const p = app.project;
  let shown = null, marker = "qcut", words = ["qcut"], timer = 0, typedFor = null, asked = 0, inFlight = Promise.resolve();
  const box = c.textarea.md({ label: "Story", rows: 14, fill: true, value: p.project ? joinStory(p.project, marker) : "", onInput: (v) => { clearTimeout(timer); typedFor = p.project && p.project.id; timer = setTimeout(() => { timer = 0; apply(v); }, 700); }, onCommit: (v) => { clearTimeout(timer); timer = 0; apply(v); } });
  const area = box.node, warn = c.hint.default({ text: "" });
  // Text typed in one project is never split into another one that was opened (by keyboard, with no blur) before the pause ran out.
  function apply(text) { if (typedFor && p.project && p.project.id !== typedFor) { typedFor = null; sync(); return inFlight; } typedFor = null; return (inFlight = split(text)); }
  async function split(text) {
    if (!p.project) return;
    const mine = ++asked, pid = p.project.id;
    try {
      const { scenes } = await app.api.storySplit(text);
      if (mine !== asked || !p.project || p.project.id !== pid) return;     // typed more meanwhile, or another project opened: not for this one
      p.edit((pr) => applyStory(pr, scenes, shown));
      if (p.project) shown = shownStory(p.project);
    } catch (err) { c.toast.warn({ text: `Could not split the story: ${err.message}` }); }
  }
  app.api.storyMarkers().then((r) => { words = r.markers || words; marker = words[0]; sync(); }).catch(() => {});
  function sync() {
    warn.setText(p.project && clash(p.project, words) ? `A scene's own text contains a cut word (${words.join(", ")}), so the story shows it as two scenes. Remove the word from that scene.` : "");
    if (!p.project || app.dom.focusOn(area) || timer) return;           // never rewrite the box while it is being typed in
    const next = joinStory(p.project, marker);
    if (area.value !== next) box.setValue(next);
    shown = shownStory(p.project);
  }
  own(app.on(sync));
  // Generate runs what the box shows: text still waiting for its pause, or still being split, lands first.
  const flush = async () => { if (timer) { clearTimeout(timer); timer = 0; apply(area.value); } await inFlight; };
  app.beforeRun.push(flush);
  own(() => { flush().finally(() => app.beforeRun.splice(app.beforeRun.indexOf(flush) >>> 0, 1)); });     // closed mid-edit: still waited for
  sync();
  return c.region.stack({ gap: "sm", fill: true, children: [
    templatesBar(app, own),
    c.toolbar.default({ items: [c.label.section({ text: "Story" })], trailing: [c.button.sm({ label: "+ Add shortcut", tone: "ghost", title: "Insert a shortcut trigger: browse the library", onClick: async () => {
      const trigger = await pickShortcut(app);
      if (!trigger) return;
      area.focus(); area.setRangeText(`${area.selectionStart && !/\s$/.test(area.value.slice(0, area.selectionStart)) ? " " : ""}${trigger} `, area.selectionStart, area.selectionEnd, "end");
      area.dispatchEvent(new Event("input", { bubbles: true }));
    } })] }),
    box, warn,
    c.hint.default({ text: `Scenes are cut at the word “${marker}”. Edits apply to the scenes as you type; the anchor is its own field.` }),
    c.collapsible.default({ label: "+ Variables", body: variablesPanel(app, own) }),
  ] });
};

const RISK = "Use at your own risk. The Helper can change almost anything in FunUI: scenes, prompts, variables, shortcuts, settings, model files and the whole pipeline. How exactly a change matches what you asked depends heavily on the model you chose, and nobody can promise that it follows instructions correctly. Read every change before you press Allow. Changes to shortcuts and to the pipeline are not covered by Undo.";

// Tabs whose backend module may be missing.
const TAB_MODULE = { enhance: "conditioning_prompt_enhancer", chat: "conditioning_prompt_enhancer" };

const sheets = { story, shortcuts, build, cuts,
  enhance, chat, files };

export default {
  id: "composer",
  mount: "timeline.status",
  needs: ["project", "api", "pipeline", "dom"],
  setup({ host, app }) {
    let win = null, owned = [];
    const cleanup = () => { owned.forEach((f) => f()); owned = []; };
    const open = () => {
      if (win) return;
      const body = c.region.stack({ gap: "sm", fill: true });
      const show = (tab) => { cleanup(); body.set([c.region.stack({ sticky: true, gap: "sm", children: [c.tabs.underline({ label: "Composer", tabs: TABS.filter((t) => !TAB_MODULE[t.value] || !app.hasModule || app.hasModule(TAB_MODULE[t.value])), value: tab, onChange: show })] }), sheets[tab](app, (off) => owned.push(off))]); };      // the tabs stay pinned at the top while a sheet scrolls
      show("story");
      win = c.floating.window({ id: "composer", title: "Composer", subtitle: "prompt craft", body, width: 420, height: 520, x: 320, y: 90, origin: () => button.node,
        onClose: () => { cleanup(); win = null; } });
    };
    const button = c.button.sm({ label: "◆ Composer", tone: "neutral", onClick: () => (win ? win.close() : open()) });
    host.append(button.node);       // the button toggles it, as v4
    // The helper changes the project itself, so it stays off until the person turns it on (Settings > Editor), with a warning.
    const helper = c.button.sm({ label: "✦ Helper", tone: "neutral", title: "Ask the model to change the project", onClick: () => { if (app.hasModule && !app.hasModule("conditioning_prompt_enhancer")) c.toast.warn({ text: "The helper needs the prompt enhancer module." }); else openHelper(app, helper); } });
    const showHelper = () => { helper.node.hidden = app.project.pref("helper", false) !== true; };
    host.append(helper.node);
    showHelper();
    const offHelper = app.on(showHelper);
    app.editorSettings.push(() => c.region.stack({ gap: "sm", children: [
      c.toggle.default({ label: "Helper", hint: "A button that opens a window where the model can change scenes, their length and the clip length when you ask. You allow each change first.",
        checked: app.project.pref("helper", false) === true, onChange: async (v) => { if (v && !(await c.modal.dialogue({ title: "Switch on the Helper?", message: RISK, tone: "danger", confirmLabel: "I understand, switch it on" }).result)) { app.project.setPref("helper", false); showHelper(); return; } app.project.setPref("helper", v); showHelper(); } }),
      c.banner.warn({ text: RISK })] }));
    app.inputHooks.push(chatHook);          // last, so a setup that failed above leaves nothing behind
    // A run that summarised the conversation says so, whether or not the Composer is open.
    let phase = null, told = new Set();
    const offRun = app.generate && app.generate.subscribe ? app.generate.subscribe((st) => {
      const was = phase; phase = st && st.phase;
      if (was && was !== phase && (phase === "done" || phase === "failed") && app.hasModule && app.hasModule("conditioning_prompt_enhancer"))
        Promise.resolve().then(() => app.api.enhancerRuns()).then((r) => {
          const hit = (r.runs || []).filter((x) => x.compacted && !told.has(x.prompt_id));
          hit.forEach((x) => told.add(x.prompt_id));
          if (hit.length) c.toast.warn({ text: "The conversation was getting long: the earlier turns were summarised for the enhancer." });
        }).catch(() => {});
    }) : () => {};
    return () => { offRun(); offHelper(); cleanup(); app.inputHooks.splice(app.inputHooks.indexOf(chatHook) >>> 0, 1); };
  },
};
