// "✦ Helper": ask the model to act on the project. Its own feature: it announces a button on the Composer's row and a switch in
// Settings > Editor, and is absent when the prompt enhancer module is. Off until the person turns it on, with a warning.
import { composer as c } from "../../composer/composer.js";
import { openHelper } from "./window.js";

const RISK = "Use at your own risk. The Helper can change almost anything in FunUI: scenes, prompts, variables, shortcuts, settings, model files and the whole pipeline. How exactly a change matches what you asked depends heavily on the model you chose, and nobody can promise that it follows instructions correctly. Read every change before you press Allow. Changes to shortcuts and to the pipeline are not covered by Undo.";

export default {
  id: "helper",
  mount: "timeline.status",
  needs: ["project", "api", "editorSettings"],
  modules: ["conditioning_prompt_enhancer"],
  setup({ host, app }) {
    const button = c.button.sm({ label: "✦ Helper", tone: "neutral", title: "Ask the model to change the project", onClick: () => openHelper(app, button) });
    const show = () => { button.node.hidden = app.project.pref("helper", false) !== true; };
    const first = () => { if (host.firstElementChild !== button.node) host.prepend(button.node); };       // the row starts with the Helper, whichever feature added its button later
    const keepFirst = typeof MutationObserver === "function" ? new MutationObserver(first) : null;
    const section = () => c.region.stack({ gap: "sm", children: [
      c.toggle.default({ label: "Helper", hint: "A button that opens a window where the model can change scenes, their length and the clip length when you ask. You allow each change first.",
        checked: app.project.pref("helper", false) === true, onChange: async (v) => {
          if (v && !(await c.modal.dialogue({ title: "Switch on the Helper?", message: RISK, tone: "danger", confirmLabel: "I understand, switch it on" }).result)) { app.project.setPref("helper", false); show(); return; }
          app.project.setPref("helper", v); show();
        } }),
      c.banner.warn({ text: RISK })] });
    show();
    first();
    if (keepFirst) keepFirst.observe(host, { childList: true });
    const off = app.on(show);
    app.editorSettings.push(section);                   // last, so a setup that failed above leaves nothing behind
    return () => { off(); if (keepFirst) keepFirst.disconnect(); app.editorSettings.splice(app.editorSettings.indexOf(section) >>> 0, 1); button.destroy(); };
  },
};
