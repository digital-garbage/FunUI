// "Scene N · Rate…": say how a render turned out. FunUI's taste learns from it on the next Generate.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { CHOICES, FORGET, nameOf, tasteOf } from "./choices.js";

/** What to say when a rating taught nothing, or "" when it did. The run captured nothing, a learner is on and no key is
 *  set: both are said. Any other reason is the server's own. */
export function notTaught(r, ps) {
  if (!(r && r.why)) return "";
  const taste = ps.modulesById && ps.modulesById().taste;
  const keyless = r.reason === "nothing" && taste && ps.useful(taste) && !String((ps.currentValues().taste || {}).key || "").trim();
  // two facts, both true, not a claimed cause: the key on screen now may not be the one the run had
  return keyless ? "Taste not taught: this run captured nothing, and no Taste key is set. Ratings teach only under a key: name one in Settings ▸ Engine ▸ System ▸ Taste key." : `Taste not taught: ${r.why}.`;
}

/** The project's scenes as menu entries, numbered as the timeline numbers them. */
export const sceneItems = (open) => (open ? open.scenes : []).map((s, i) => ({ id: s.id, label: `Scene ${i + 1}`, hint: (s.text || "").slice(0, 24) }));

export default {
  id: "rating",
  mount: "timeline.actions",
  needs: ["project", "api", "selection", "pipeline"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host);
    const target = () => {                                  // the picked clip's generation unit, when it has a render to talk about
      const open = p.project, sc = open && open.scenes.find((s) => s.id === sel.focus);
      if (!sc || !isGenerative(sc)) return null;
      const root = unitRoot(open, genUnitId(sc));
      const render = root && (open.scene_renders || {})[root.id];
      return render && render.media ? { root, render, no: open.scenes.indexOf(sc) + 1 } : null;
    };
    const rate = async (value) => {
      const t = target();
      if (!t) return;
      // What was rated and when: the prompt that made the render (the scene's text, for a render that did not keep it); when, for a run of dislikes.
      p.edit((pr) => { const s = pr.scenes.find((x) => x.id === t.root.id); if (s) { s.rating = value; s.rated = nameOf(value) ? { text: t.render.text ?? s.text ?? "", at: Date.now() } : {}; } return s; });
      const taste = tasteOf(value);
      if (!t.render.promptId || taste === null) return;      // no run to pair it with, or a word that teaches nothing: it stays a label
      const body = taste === "clear" ? { rating: null, axis: null } : taste;
      try {
        const r = await app.api.rateTaste(t.render.promptId, body.rating, body.axis);
        const text = notTaught(r, app.pipeline);
        if (text) c.toast.warn({ text });
      } catch { /* no taste module here: the rating stays a label */ }
    };
    const draw = () => {
      const t = target(), open = p.project, focus = open ? open.scenes.findIndex((s) => s.id === sel.focus) : -1;
      if (!t) return put();
      const picker = c.button.sm({ label: `Scene ${focus + 1} ▾`, tone: "ghost", title: "Pick another scene: the same as picking its clip on the timeline", onClick: () => {
        const items = sceneItems(p.project);
        c.menu.dropdown({ anchor: picker, items, onPick: (id) => sel.pick(id) });
      } });
      const now = nameOf(t.root.rating);
      const button = c.button.sm({ label: now ? `★ ${now}` : "Rate scene…", tone: now ? "neutral" : "ghost", title: "Rate this scene's render: features that learn from ratings use it on the next generation",
        onClick: async () => {
          const pick = await c.modal.choice({ title: "How did it turn out?", items: [...CHOICES.map((o) => ({ id: o.value, label: o.label, hint: o.hint })), { id: FORGET, label: "Forget my rating", hint: "Clears it: nothing is learned from this clip." }] }).result;
          if (pick) rate(pick);
        } });
      const note = String(p.pref("rating_comment", "") || "");
      const comment = c.button.sm({ label: note ? "✎ Comment" : "Comment…", tone: note ? "neutral" : "ghost", title: note ? `Sent with the next generation's rating line, then cleared: ${note}` : "Write a note that goes after the rating in the next generation's enhancer request (needs Pass the rating in Composer > Enhance); it is cleared after that run",
        onClick: async () => {
          const text = await c.modal.prompt({ title: "Comment on the rating", label: "Sent once, with the next generation", value: note, placeholder: "e.g. the motion was too fast", confirmLabel: "Save" }).result;
          if (text !== null && text !== undefined) { p.setPref("rating_comment", text.trim()); draw(); }
        } });
      put(picker.node, button.node, comment.node);
    };
    draw();
    return app.on(draw);
  },
};
