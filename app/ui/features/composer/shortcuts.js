// Shortcuts: the library of trigger -> replacement phrases, expanded into a prompt when it is generated.
import { composer as c } from "../../composer/composer.js";

const lines = (text) => String(text || "").split("\n").map((l) => l.trim()).filter(Boolean);

export function shortcuts(app, own) {
  const page = c.region.stack({ gap: "sm" });
  let alive = true, box = null;
  own(() => { alive = false; });
  const tell = (text) => c.toast.warn({ text });

  function edit(existing) {
    const draft = { name: "", triggers: [], replacements: [], enabled: true, category: "", sub_category: "", ...(existing || {}) };
    const field = (label, control, hint) => c.field.default({ label, hint, control });
    const win = c.modal.generic({ title: existing ? `Edit “${existing.name}”` : "New shortcut", size: "md", body: c.region.stack({ gap: "sm", children: [
      field("Name", c.input.md({ label: "Name", value: draft.name, onInput: (v) => { draft.name = v; } })),
      field("Triggers", c.textarea.md({ label: "Triggers", rows: 3, value: draft.triggers.join("\n"), onInput: (v) => { draft.triggers = lines(v); } }), "One word or phrase per line. Any of them fires the shortcut."),
      field("Replacements", c.textarea.md({ label: "Replacements", rows: 5, value: draft.replacements.join("\n"), onInput: (v) => { draft.replacements = lines(v); } }), "One per line; with several, one is drawn each time."),
      c.field.row({ fields: [field("Category", c.input.md({ label: "Category", value: draft.category, onInput: (v) => { draft.category = v; } })), field("Sub-category", c.input.md({ label: "Sub-category", value: draft.sub_category, onInput: (v) => { draft.sub_category = v; } }))] }),
      c.toggle.default({ label: "Enabled", checked: draft.enabled, onChange: (v) => { draft.enabled = v; } }),
    ] }) });
    const save = async () => {
      try { draw(await app.api.saveShortcut(draft, existing && existing.name)); win.close("done"); } catch (err) { tell(err.message); }
    };
    const remove = async () => {
      if (!(await c.modal.dialogue({ title: "Delete shortcut", message: `Delete “${existing.name}”?`, tone: "danger", confirmLabel: "Delete" }).result)) return;
      try { draw(await app.api.deleteShortcut(existing.name)); win.close("done"); } catch (err) { tell(err.message); }
    };
    win.setFooter({ actions: [existing ? c.button.sm({ label: "Delete", tone: "danger", onClick: remove }) : null, c.button.sm({ label: "Save", tone: "primary", onClick: save })].filter(Boolean) });
  }

  /** Phrases that repeat across shortcuts; picking one hands it to `onPick`. */
  async function phrasesWindow(onPick) {
    let phrases, total = 0;
    try { const got = await app.api.shortcutPhrases(); phrases = got.phrases || []; total = got.total || phrases.length; } catch (err) { return tell(err.message); }
    const win = c.modal.stacked({ title: "Common phrases", subtitle: `Words and phrases used in more than one shortcut, longest shared run first within the same count.${total > phrases.length ? ` Showing the ${phrases.length} most shared of ${total}.` : ""} Pick one to look at it or change it everywhere.`, size: "md", body: c.filterList.md({
      items: phrases.map((p) => ({ id: p.phrase, label: p.phrase, hint: `${p.shortcuts} shortcuts, ${p.uses} ${p.uses === 1 ? "line" : "lines"}` })),
      placeholder: "Search phrases", empty: "No phrase is used in more than one shortcut.",
      onChange: (phrase) => { win.close("picked"); onPick(phrase); } }) });
  }

  /** Find a phrase in the text of every shortcut, or replace it in all of them at once. */
  function findWindow(start = "", mode = "find") {
    const state = { find: start, replace: "", matchCase: false, wholeWords: true, mode };
    const results = c.region.stack({ gap: "sm" });
    let ticket = 0, timer = null, apply = null;
    const findBox = c.input.md({ label: "Find", value: start, placeholder: "Find", onInput: (v) => { state.find = v; soon(); } });
    const replaceBox = c.input.md({ label: "Replace with", value: "", placeholder: "Replace with", onInput: (v) => { state.replace = v; soon(); } });
    const findField = c.field.default({ label: "Find", control: findBox });
    const replaceField = c.field.default({ label: "Replace with", hint: "Leave it empty to remove the phrase.", control: replaceBox });
    const fields = c.region.stack({ gap: "sm", children: state.mode === "replace" ? [findField, replaceField] : [findField] });
    let found = 0;
    const sync = () => { apply.setLabel(state.replace === "" ? "Remove everywhere" : "Replace everywhere"); apply.setDisabled(state.mode !== "replace" || !found); };
    const note = (text) => { found = 0; results.set([c.hint.default({ text })]); sync(); };

    async function look() {
      const mine = ++ticket;
      if (!state.find.trim()) return note("Type the phrase to look for. It is matched exactly as typed.");
      try {
        const asked = state.mode === "replace" ? await app.api.replaceShortcuts(state.find, state.replace, state.matchCase, false, state.wholeWords) : await app.api.findShortcuts(state.find, state.matchCase, state.wholeWords);
        if (mine !== ticket || !alive) return;
        const rows = asked.matches || [], names = new Set(rows.map((m) => m.name));
        if (!rows.length) return note("No shortcut contains that phrase.");
        found = rows.length;
        const total = rows.reduce((n, m) => n + m.count, 0);
        results.set([
          c.hint.default({ text: `${total} ${total === 1 ? "match" : "matches"} in ${names.size} ${names.size === 1 ? "shortcut" : "shortcuts"}.${(asked.skipped || []).length ? ` ${asked.skipped.length} would be left as they are, because nothing would remain of them.` : ""}` }),
          c.filterList.md({ items: rows.map((m) => ({ id: `${m.name}#${m.index}`, label: m.name, hint: state.mode === "replace" ? `${m.before} → ${m.after}` : m.before })), placeholder: "Search the matches", empty: "Nothing matches." }),
        ]);
        sync();
      } catch (err) { if (mine === ticket) note(err.message); }
    }
    const soon = () => { clearTimeout(timer); timer = setTimeout(look, 200); };
    own(() => clearTimeout(timer));

    async function commit() {
      clearTimeout(timer);
      await look();                                 // the count asked about is the one for what is typed now
      if (!found) return;
      const ok = await c.modal.dialogue({ title: state.replace === "" ? "Remove everywhere" : "Replace everywhere", tone: "danger",
        message: `Change ${found} ${found === 1 ? "line" : "lines"} in your saved shortcuts? This cannot be undone.`, confirmLabel: state.replace === "" ? "Remove" : "Replace" }).result;
      if (!ok) return;
      try {
        const done = await app.api.replaceShortcuts(state.find, state.replace, state.matchCase, true, state.wholeWords);
        draw(done);
        c.toast.good({ text: `Changed ${done.changed} ${done.changed === 1 ? "shortcut" : "shortcuts"}.` });
        win.close("done");
      } catch (err) { tell(err.message); }
    }

    const win = c.modal.generic({ title: "Find and replace", subtitle: "Searches the replacement text of every shortcut, paused ones too.", size: "lg", body: c.region.stack({ gap: "sm", children: [
      c.segmented.md({ label: "Mode", value: state.mode, options: [{ value: "find", label: "Find" }, { value: "replace", label: "Replace" }], onChange: (v) => { state.mode = v; fields.set(v === "replace" ? [findField, replaceField] : [findField]); look(); } }),
      fields,
      c.toggle.default({ label: "Match upper and lower case", checked: false, onChange: (v) => { state.matchCase = v; look(); } }),
      c.toggle.default({ label: "Whole words only", hint: "Off also finds the phrase inside longer words, such as “cat” in “scatter”.", checked: true, onChange: (v) => { state.wholeWords = v; look(); } }),
      c.button.sm({ label: "Common phrases…", onClick: () => phrasesWindow((phrase) => { findBox.setValue(phrase); state.find = phrase; look(); }) }),
      results,
    ] }) });
    apply = c.button.sm({ label: "Replace everywhere", tone: "primary", disabled: true, onClick: commit });
    win.setFooter({ actions: [apply] });
    look();
  }

  /** Export / import / new category / delete all. */
  function tools(anchor, lib) {
    const did = (p) => p.then(draw).catch((err) => tell(err.message));
    c.menu.dropdown({ anchor: anchor || app.dom.body, items: [{ id: "find", label: "Find and replace…", disabled: !(lib.shortcuts || []).length }, { id: "phrases", label: "Common phrases…", disabled: !(lib.shortcuts || []).length }, { separator: true }, { id: "export", label: "Export all…" }, { id: "import", label: "Import…" }, { id: "category", label: "New category…" }, { separator: true }, { id: "clear", label: "Delete all", danger: true, disabled: !(lib.shortcuts || []).length }],
      onPick: async (id) => {
        if (id === "find") return findWindow();
        if (id === "phrases") return phrasesWindow((phrase) => findWindow(phrase, "replace"));
        if (id === "export") return app.dom.download("/fun/api/shortcuts/export", "funui_shortcuts.json");
        if (id === "import") {
          const picked = async (file) => {
            if (!file) return;
            let data;
            try { data = JSON.parse(await file.text()); } catch { return tell("That file is not valid JSON."); }
            c.modal.choice({ title: "Import shortcuts", subtitle: "Merge them into your library, or replace the library with them?",
              items: [{ id: "merge", label: "Merge into my library" }, { id: "replace", label: "Replace my library (deletes the shortcuts you have now)" }],
              onPick: (mode) => did(app.api.importShortcuts(data, mode)) });       // closing the window imports nothing
          };
          return picked(await app.dom.pickFile(".json,application/json"));
        }
        if (id === "category") {
          const category = await c.modal.prompt({ title: "New category", label: "Category", confirmLabel: "Next" }).result;
          if (!category || !category.trim()) return;
          const sub = await c.modal.prompt({ title: "Sub-category", label: "Sub-category (optional)", confirmLabel: "Add" }).result;
          if (sub === null || sub === undefined) return;
          return did(app.api.addShortcutCategory(category.trim(), (sub || "").trim()));
        }
        if (await c.modal.dialogue({ title: "Delete all shortcuts", message: "Delete every shortcut in the library? Export first if you might want them back.", tone: "danger", confirmLabel: "Delete all" }).result) did(app.api.clearShortcuts());
      } });
  }

  function draw(lib) {
    if (!alive) return;
    const items = [...(lib.shortcuts || [])].sort((x, y) => (x.category || "").localeCompare(y.category || "") || x.name.localeCompare(y.name))       // by name: the pause mark in a label must not decide the order
      .map((s) => ({ id: s.name, label: `${s.enabled ? "" : "⏸ "}${s.name}`, hint: s.triggers.join(", "), keywords: `${s.category} ${s.sub_category} ${s.replacements.join(" ")}`, group: s.category || "" }));
    const query = box ? box.query : "";                    // what was typed in the search box survives a save, a delete, an import
    box = c.filterList.md({ items, query, placeholder: "Search shortcuts", empty: "No shortcuts yet. Add one with + New.", onChange: (id) => edit((lib.shortcuts || []).find((s) => s.name === id)) });
    const more = c.button.sm({ label: "⋯", tone: "ghost", title: "Export, import, categories, delete all", onClick: () => tools(more.node, lib) });
    page.set([
      c.toolbar.default({ items: [c.label.section({ text: "Shortcuts" })], trailing: [c.button.sm({ label: "+ New", tone: "primary", onClick: () => edit(null) }), more] }),
      box,
      c.hint.default({ text: "Type a trigger in any prompt and it is replaced when you generate." }),
    ]);
  }
  app.api.shortcuts().then(draw).catch((err) => page.set([c.banner.warn({ text: `Could not load shortcuts: ${err.message}` })]));
  return page;
}
