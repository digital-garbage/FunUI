// What sits around the Story box: saved templates (the scenes), the project's $variables list, and the shortcut picker.
import { composer as c } from "../../composer/composer.js";
import { applyStory, storyRoots } from "./story.js";

const clone = (x) => JSON.parse(JSON.stringify(x));
const NONE = "__none__";

/** Templates: pick one to apply, Save snapshots the Story, rename/delete act on the applied one. A region that redraws itself with the project. */
export function templatesBar(app, own) {
  const p = app.project, bar = c.region.stack({ gap: "xs" });
  const ask = (title, value) => c.modal.prompt({ title, label: "Name", value, confirmLabel: "Save" }).result;
  const tpls = () => (p.project && p.project.prompt_templates) || [];
  const save = async () => {
    const open = p.project, active = open.active_prompt_template || "";
    const name = ((await ask("Template name", tpls().some((t) => t.name === active) ? active : "")) || "").trim();
    if (!name) return;
    p.edit((pr) => {
      const snap = { name, anchor: pr.anchor || "", scenes: storyRoots(pr).map((r) => (r.text || "").trim()) };
      pr.prompt_templates = [...(pr.prompt_templates || []).filter((t) => t.name !== name), snap];
      pr.active_prompt_template = name;
      return true;
    });
  };
  const apply = (name) => p.edit((pr) => {
    const tpl = (pr.prompt_templates || []).find((t) => t.name === name);
    if (!tpl) return false;
    pr.active_prompt_template = name;
    if (typeof tpl.anchor === "string") pr.anchor = tpl.anchor;
    applyStory(pr, Array.isArray(tpl.scenes) ? tpl.scenes : [tpl.prompt || ""]);      // an older template is one prompt: it applies as one scene
    return true;
  });
  const clear = async () => {
    if (!(await c.modal.dialogue({ title: "Clear the Story", message: "Empties the anchor and every scene's text, and stops using the current template. Variables and postfix are left alone. Undo restores it.", confirmLabel: "Clear", tone: "danger" }).result)) return;
    p.edit((pr) => { pr.anchor = ""; storyRoots(pr).forEach((s) => { s.text = ""; }); pr.active_prompt_template = ""; return true; });
  };
  const rename = async (active) => {
    const next = ((await ask("Rename template", active)) || "").trim();
    if (!next || next === active) return;
    if (tpls().some((t) => t.name === next)) return c.toast.warn({ text: `A template called “${next}” already exists.` });
    p.edit((pr) => { pr.prompt_templates.forEach((t) => { if (t.name === active) t.name = next; }); pr.active_prompt_template = next; return true; });
  };
  const remove = async (active) => {
    if (!(await c.modal.dialogue({ title: "Delete template", message: `Delete “${active}”? The scenes stay as they are.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
    p.edit((pr) => { pr.prompt_templates = pr.prompt_templates.filter((t) => t.name !== active); pr.active_prompt_template = ""; return true; });
  };
  let key = "";
  const draw = () => {
    const open = p.project, active = open ? open.active_prompt_template || "" : "", known = tpls().some((t) => t.name === active);
    const next = JSON.stringify([open && open.id, active, tpls().map((t) => t.name)]);
    if (next === key) return;
    key = next;
    bar.set([c.toolbar.default({
      items: [c.select.sm({ label: "Templates", disabled: !open, value: known ? active : "",
        options: [{ value: "", label: "Templates…" }, { value: NONE, label: "— None (clear all prompts) —" }, ...tpls().map((t) => ({ value: t.name, label: t.name }))],
        onChange: async (v) => { if (v === NONE) await clear(); else if (v) apply(v); key = ""; draw(); } })],
      trailing: [c.button.sm({ label: known ? "Update…" : "Save", tone: "ghost", disabled: !open, title: known ? `Overwrite “${active}” or save under a new name` : "Save the Story as a template", onClick: save }),
        ...(known ? [c.button.sm({ label: "✎", tone: "ghost", title: `Rename “${active}”`, onClick: () => rename(active) }), c.button.sm({ label: "✕", tone: "ghost", title: `Delete “${active}”`, onClick: () => remove(active) })] : [])] })]);
  };
  draw();
  own(app.on(draw));
  return bar;
}

/** `$name` → text, filled in at generation. One tab per variable; its value box is a fixed height that scrolls, and each tab keeps its own caret and scroll while you visit others. Rows commit on leaving the field; empty names are dropped on save. */
export function variablesPanel(app, own) {
  const p = app.project, page = c.region.stack({ gap: "xs" });
  let shown = "", active = 0, pid = null;
  const view = {};                                   // tab index -> { s, e, top } of its value box
  const rows = () => (p.project && p.project.variables) || [];
  const box = () => page.node.querySelector("textarea");
  const remember = () => { const t = box(); if (t) view[active] = { s: t.selectionStart, e: t.selectionEnd, top: t.scrollTop }; };
  const write = (list) => p.setField("variables", list.map((v) => ({ name: String(v.name || "").replace(/^\$+/, "").trim(), value: String(v.value || "") })).filter((v) => v.name || v.value));
  const draw = (focus) => {
    if (pid !== (p.project && p.project.id)) { pid = p.project && p.project.id; active = 0; for (const k in view) delete view[k]; }
    shown = JSON.stringify([pid, rows()]);
    active = Math.max(0, Math.min(active, rows().length - 1));
    const v = rows()[active];
    page.set([
      ...(v ? [c.tabs.underline({ label: "Variables", tabs: rows().map((r, i) => ({ value: String(i), label: r.name || "(unnamed)" })), value: String(active), onChange: (x) => { remember(); active = Number(x); draw(true); } }),
        c.toolbar.default({
          items: [c.input.sm({ label: "Name", value: v.name, placeholder: "name", onCommit: (x) => { const l = clone(rows()); l[active].name = x; write(l); } }),
            c.textarea.md({ label: "Value", rows: 6, value: v.value, placeholder: "text", onCommit: (x) => { const l = clone(rows()); l[active].value = x; write(l); } })],
          trailing: [c.button.sm({ label: "✕", tone: "ghost", title: "Remove", onClick: () => { const l = rows().filter((_, k) => k !== active); delete view[active]; write(l); draw(); } })] })] : []),
      c.button.sm({ label: "+ Add variable", tone: "ghost", disabled: !p.project, onClick: () => { remember(); write([...rows(), { name: "name", value: "" }]); active = rows().length - 1; draw(); } }),
      c.hint.default({ text: "Write $name in a prompt; it becomes the variable's text when you generate." }),
    ]);
    const t = box(), st = view[active];
    if (t) {
      if (st) { t.setSelectionRange(st.s, st.e); t.scrollTop = st.top; }
      if (focus) t.focus({ preventScroll: true });
      t.addEventListener("scroll", remember); t.addEventListener("blur", remember); t.addEventListener("select", remember);
    }
  };
  draw();
  own(app.on(() => { const typing = app.dom.typingIn(page.node); if (!typing && shown !== JSON.stringify([p.project && p.project.id, rows()])) draw(); }));
  return page;
}

/** A window of the shortcut library; picking one gives back its first trigger. */
export async function pickShortcut(app) {
  let lib; try { lib = await app.api.shortcuts(); } catch (err) { c.toast.warn({ text: err.message }); return null; }
  const all = (lib.shortcuts || []).filter((s) => (s.triggers || []).length);
  if (!all.length) { c.toast.warn({ text: "No shortcuts yet. Add some in the Shortcuts tab." }); return null; }
  return new Promise((resolve) => {
    const items = all.map((s) => ({ id: s.name, label: s.name, hint: (s.replacements || [])[0] || "", keywords: `${s.triggers.join(" ")} ${s.category} ${s.sub_category}`, group: [s.category, s.sub_category].filter(Boolean).join(" › ") }))
      .sort((x, y) => String(x.group).localeCompare(String(y.group)) || String(x.label).localeCompare(String(y.label)));
    let chosen = null;
    const win = c.modal.generic({ title: "Add shortcut", size: "md", onClose: () => resolve(chosen),
      body: c.filterList.md({ items, placeholder: "Search shortcuts", empty: "Nothing matches.", onChange: (id) => { chosen = all.find((s) => s.name === id).triggers[0]; win.close("done"); } }) });
  });
}
