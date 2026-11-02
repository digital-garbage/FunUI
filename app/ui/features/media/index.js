// The media bin: files brought in (reference images, clips), kept apart from what runs produce.
import { composer as c } from "../../composer/composer.js";
import { learn } from "../../shell/bin.js";
import { MAX_REFERENCES, referenceNumbers, sceneReferences } from "../../shell/reference_wiring.js";
import { MEDIA_DRAG } from "../../shell/dnd.js";
import { allReferences, clearReferences } from "../../shell/edits.js";

const base = (id) => `/fun/api/media/${encodeURIComponent(id)}`;

export default {
  id: "media",
  mount: "assets.media",
  needs: ["project", "api", "dom"],
  setup({ host, app }) {
    const p = app.project, api = app.api;
    let items = [];
    const tell = (text) => c.toast.warn({ text });

    const toggle = (id) => { if (chosen.has(id)) chosen.delete(id); else chosen.add(id); draw(true); };
    const FILTERS = [{ value: "all", label: "All" }, { value: "video", label: "Video" }, { value: "audio", label: "Audio" }, { value: "image", label: "Images" }];
    const SORTS = [{ value: "name", label: "Name A-Z" }, { value: "name-", label: "Name Z-A" }, { value: "kind", label: "Type" }, { value: "added", label: "Date added" }];
    const VIEWS = [{ value: "adaptive", label: "Grid" }, { value: "list", label: "List" }, { value: "icons", label: "Icons" }];
    const DENSITY = [{ value: "0", label: "Auto" }, ...[1, 2, 3, 4].map((n) => ({ value: String(n), label: `${n}×` }))];
    const show = { filter: "all", sort: "name", view: "adaptive", cols: "0" };
    // How the bin is laid out is remembered with the project (a rental's browser starts empty), checked against what is offered.
    // The project may open after the bin does, so it is read whenever a different project shows up, not once at setup.
    const KEPT = { sort: SORTS, view: VIEWS, cols: DENSITY };
    let adopted = null;
    const remember = () => { if (p.setPref) p.setPref("media_show", { sort: show.sort, view: show.view, cols: show.cols }); };

    let selecting = false;
    const chosen = new Set();
    const peek = (it) => { const item = items.find((m) => m.id === it.id); if (!item) return; app.mediaPeek.item = item; app.say("media.peek"); };       // a click looks at it on the monitor
    const pick = (it) => {                                  // an image becomes the selected scene's aspect ratio source
      const sc = p.selected, item = items.find((m) => m.id === it.id);
      if (!sc || !item) return tell("Select a scene first.");
      if (item.kind !== "image") return tell("Only an image can be a scene's aspect ratio source.");
      p.setScene(sc.id, "source_image", item.id);
      draw();
    };
    const inform = (text) => c.toast.info({ text });
    /** Pin a picture as the aspect ratio source of every scene, and of each scene added later; or take the pin off (scenes keep what they have). */
    const resAll = (it) => {
      const item = items.find((m) => m.id === it.id);
      if (!item || item.kind !== "image") return tell("Only an image can be a scene's aspect ratio source.");
      if ((p.project || {}).fixed_source_image === item.id) {
        p.edit((pr) => { pr.fixed_source_image = null; return true; });
        inform("Unpinned. Scenes keep the aspect ratio source they have now.");
      } else {
        let n = 0, all = 0;
        p.edit((pr) => { pr.fixed_source_image = item.id; for (const s of pr.scenes || []) { all += 1; if (s.source_image !== item.id) { s.source_image = item.id; n += 1; } } return true; });
        inform(`Pinned as the aspect ratio source for all scenes (${n} of ${all} changed). New scenes start with it.`);
      }
      draw(true);
    };
    /** Every scene gets this as a reference. A scene that already has it (also as a pinned one) or has no free slot is skipped; what a model can take is checked when the run is made. */
    const refAll = (it) => {
      let n = 0, have = 0, full = 0;
      p.edit((pr) => {
        const fx = pr.fixed_references || {};
        for (const s of pr.scenes || []) {
          const plan = sceneReferences(s, fx);
          if (plan.ids.includes(it.id)) { have += 1; continue; }
          if (plan.ids.length >= MAX_REFERENCES) { full += 1; continue; }
          s.references = [...(s.references || []), it.id]; n += 1;
        }
        return n > 0;
      });
      inform(`Added as a reference to ${n} scene${n === 1 ? "" : "s"}.${have ? ` ${have} already had it.` : ""}${full ? ` ${full} had no free reference slot.` : ""}`);
      draw();
    };
    /** The slot number this picture holds in the selected scene, and whether it is pinned there: null when it is not one of its references. */
    const slotOf = (id) => {
      const sc = p.selected;
      if (!sc) return null;
      const plan = sceneReferences(sc, (p.project || {}).fixed_references), i = plan.ids.indexOf(id);
      return i < 0 ? null : { n: referenceNumbers(plan.ids, plan.pins)[i], pinned: id in plan.pins };
    };
    const pinItem = (it) => { const s = slotOf(it.id); return s && s.n <= 16 ? [{ id: "pinref", label: s.pinned ? `Unpin reference from slot R${s.n}` : `Pin reference to slot R${s.n}` }] : []; };
    const extras = (it) => (app.mediaMenu || []).flatMap((f) => { try { return f(it, items.find((m) => m.id === it.id)) || []; } catch { return []; } });       // entries other features add to a tile's menu: { id, label, run }
    const context = (it, e) => c.menu.context({ x: e.clientX, y: e.clientY, onPick: (id) => (act[id] ? act[id](it) : extras(it).find((x) => x.id === id)?.run()),
      items: [{ id: "look", label: "Look at it on the monitor" }, ...(p.selected && p.selected.source_image === it.id ? [{ id: "unres", label: "Stop using as the selected scene's aspect ratio source" }]
          : [{ id: "res", label: "Use as the selected scene's aspect ratio source", disabled: !p.selected || it.kind === "audio" || !items.some((m) => m.id === it.id && m.kind === "image") }]),
        { id: "resall", label: (p.project || {}).fixed_source_image === it.id ? "Unpin as the aspect ratio source for all scenes" : "Pin as the aspect ratio source for all scenes", disabled: !items.some((m) => m.id === it.id && m.kind === "image") },
        { id: "refall", label: "Use as a reference for all scenes" },
        p.selected && (p.selected.references || []).includes(it.id) ? { id: "unref", label: "Stop using as a reference for the selected scene" }
          : { id: "ref", label: "Use as a reference for the selected scene", disabled: !p.selected || !!(slotOf(it.id) || {}).pinned }, ...pinItem(it), { id: "rename", label: "Rename…" },
        { id: "export", label: "Save to your computer", disabled: !items.some((m) => m.id === it.id && m.kind !== "audio") }, ...extras(it).map(({ id, label, disabled }) => ({ id, label, disabled })), { separator: true }, { id: "delete", label: "Delete from the bin", danger: true }] });
    /** Delete these from the bin, and nothing in this project may keep pointing at them. */
    async function gone(ids) {
      const dead = new Set();
      try {
        for (const id of ids) { await api.deleteMedia(id); dead.add(id); }
      } catch (err) { tell(`${err.message}${dead.size ? ` (${dead.size} of ${ids.length} were deleted)` : ""}`); }
      try {
        p.edit((pr) => {
          if (dead.has(pr.fixed_source_image)) pr.fixed_source_image = null;
          const fx = pr.fixed_references || {}, stale = Object.keys(fx).filter((id) => dead.has(id));
          if (stale.length) pr.fixed_references = Object.fromEntries(Object.entries(fx).filter(([id]) => !dead.has(id)));
          return stale.length > 0 | pr.scenes.reduce((hit, s) => {
          const had = dead.has(s.source_image) || (s.references || []).some((r) => dead.has(r));
          if (dead.has(s.source_image)) s.source_image = "";
          if (had) s.references = (s.references || []).filter((r) => !dead.has(r));
          return hit || had;
        }, false);
        });
        ids.forEach((id) => chosen.delete(id));
      } catch (err) { tell(err.message); }
      await refresh();
    }
    const act = {
      look: (it) => peek(it), res: (it) => pick(it), resall: resAll, refall: refAll,
      ref: (it) => p.setScene(p.selected.id, "references", [...(p.selected.references || []), it.id]),
      pinref: (it) => {                                  // pin at the number it holds now, for every scene, or take the pin off
        const s = slotOf(it.id);
        if (s) p.edit((pr) => { const next = { ...(pr.fixed_references || {}) }; if (s.pinned) delete next[it.id]; else next[it.id] = s.n; pr.fixed_references = next; return true; });
      },
      unref: (it) => p.setScene(p.selected.id, "references", (p.selected.references || []).filter((r) => r !== it.id)),
      unres: () => p.setScene(p.selected.id, "source_image", ""),
      rename: async (it) => {
        const name = await c.modal.prompt({ title: "Rename media", label: "Name", value: (items.find((m) => m.id === it.id) || {}).name || "", confirmLabel: "Rename" }).result;
        if (!name || !name.trim()) return;
        try { await api.renameMedia(it.id, name.trim()); await refresh(); } catch (err) { tell(err.message); }
      },
      export: (it) => { const m = items.find((x) => x.id === it.id); app.dom.download(`${base(it.id)}/file`, m ? m.name : it.id); },
      delete: async (it) => {
        const name = (items.find((m) => m.id === it.id) || {}).name || "this file";
        if (!(await c.modal.dialogue({ title: "Delete media", message: `Delete “${name}” from the bin for good? Scenes in other projects that use it will lose it too.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
        await gone([it.id]);
      },
    };
    const props = { id: "media", items: [], empty: "No media yet. Drop images or clips here.", onActivate: (it) => (selecting ? toggle(it.id) : peek(it)), onContext: context, drag: { type: MEDIA_DRAG, data: (it) => ({ id: it.id, kind: (items.find((m) => m.id === it.id) || {}).kind }) } };
    const galleries = { adaptive: c.gallery.adaptive(props), list: c.gallery.list(props), icons: c.gallery.icons(props) };
    const shelf = c.region.stack({ gap: "none", children: [galleries.adaptive] });
    const note = c.text.sm({ text: "" });
    const drop = c.dropzone.default({ label: "Drop or choose files", hint: "images, clips, audio",
      onFiles: async (files) => {
        try { const r = await api.uploadMedia(files, (i, n, name, f) => note.setText(`Uploading ${i + 1}/${n}: ${name}${f ? ` ${f >= 1 ? "(saving...)" : `${Math.round(f * 100)}%`}` : ""}`)); (r.problems || []).forEach(tell); await refresh(); } catch (err) { tell(err.message); }
        note.setText("");
      } });
    const seg = (label, options, key) => c.segmented.sm({ label, options, value: show[key], onChange: (v) => { show[key] = v; remember(); draw(true); } });
    const tileSize = seg("Tile size", DENSITY, "cols"), view = seg("View", VIEWS, "view");       // only the grid has columns
    const sort = c.select.sm({ label: "Sort", options: SORTS, value: show.sort, onChange: (v) => { show.sort = v; remember(); draw(true); } });
    const selectBtn = c.button.sm({ label: "Select", tone: "ghost", title: "Pick several files, then delete them together", onClick: () => { selecting = !selecting; chosen.clear(); draw(true); } });
    const bulk = c.button.sm({ label: "Delete (0)", tone: "danger", onClick: async () => {
      const ids = [...chosen];
      if (!ids.length || !(await c.modal.dialogue({ title: "Delete media", message: `Delete ${ids.length} file${ids.length > 1 ? "s" : ""} from the bin for good? Scenes in other projects that use them will lose them too.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
      await gone(ids);
    } });
    const controls = { sort, view: view, cols: tileSize };
    bulk.node.hidden = true;
    const clearRefs = c.button.sm({ label: "Clear references", tone: "ghost", title: "Stop using every reference picture in every scene (pinned ones stay)", onClick: () => p.edit(clearReferences) });
    host.append(c.toolbar.default({ items: [c.text.sm({ text: "Media" })], trailing: [clearRefs, selectBtn, bulk] }).node, drop.node, note.node, seg("Show", FILTERS, "filter").node, sort.node,
      view.node, tileSize.node, shelf.node);

    let drawn = "";
    function adopt() {
      const id = p.project ? p.project.id : null;
      if (!id || id === adopted) return false;
      adopted = id;
      const kept = p.pref ? p.pref("media_show", null) : null;
      if (kept && typeof kept === "object") for (const k in KEPT) if (KEPT[k].some((o) => o.value === kept[k])) { show[k] = kept[k]; controls[k].setValue(kept[k]); }
      return true;
    }
    function draw(force) {
      if (adopt()) force = true;
      const sc = p.selected, plan = sc ? sceneReferences(sc, (p.project || {}).fixed_references) : { ids: [], pins: {} }, refs = plan.ids, nums = referenceNumbers(refs, plan.pins);
      const key = `${allReferences(p.project || {}).size}|${(sc || {}).source_image}|${refs}|${nums}|${JSON.stringify(plan.pins)}|${items.map((m) => m.id)}|${selecting}|${[...chosen]}|${show.filter}`;
      if (!force && key === drawn) return;           // typing elsewhere must not rebuild the thumbnails
      drawn = key;
      const visible = new Set(items.filter((m) => show.filter === "all" || m.kind === show.filter).map((m) => m.id));
      for (const id of [...chosen]) if (!visible.has(id)) chosen.delete(id);       // a file out of sight is not picked
      const rev = show.sort.endsWith("-") ? -1 : 1, by = show.sort.replace("-", "");
      const shown = items.filter((m) => show.filter === "all" || m.kind === show.filter)
        .sort((x, y) => rev * (by === "added" ? (y.added || 0) - (x.added || 0) : String(x[by === "kind" ? "kind" : "name"]).localeCompare(String(y[by === "kind" ? "kind" : "name"]))));
      const g = galleries[show.view];
      shelf.set([g]);
      if (g.setCols) g.setCols(Number(show.cols));
      tileSize.node.hidden = !g.setCols;
      // what the selected scene uses is marked on its tile, so a reference can be told from the rest
      const flags = (m) => [sc && sc.source_image === m.id ? { text: "◐", title: "The selected scene's aspect ratio source" } : null,
        refs.includes(m.id) ? { text: `${plan.pins[m.id] ? "📌" : ""}R${nums[refs.indexOf(m.id)]}`, title: `Reference slot ${nums[refs.indexOf(m.id)]} for the selected scene${plan.pins[m.id] ? ", pinned to every scene" : ""}` } : null].filter(Boolean);
      g.setItems(shown.map((m) => ({ id: m.id, label: m.name, thumb: m.kind === "audio" ? undefined : `${base(m.id)}/thumb`, badge: m.kind, flags: flags(m) })));
      g.setValue(selecting ? [...chosen] : sc ? [sc.source_image, ...refs].filter(Boolean) : []);
      selectBtn.setLabel(selecting ? "Done" : "Select");
      bulk.node.hidden = !selecting;
      bulk.setLabel(`Delete (${chosen.size})`);
      bulk.setDisabled(!chosen.size);
      const inUse = allReferences(p.project || {}).size;
      clearRefs.node.hidden = !inUse;
      clearRefs.setLabel(`Clear references (${inUse})`);
    }
    async function refresh() {
      try {
        items = (await api.media()).media || []; learn(items); app.say("bin");
      } catch (err) { tell(err.message); }
      draw(true);
    }
    refresh();
    return app.on((what) => (what === "media" ? refresh() : draw()));       // "media": another feature put something in the bin
  },
};
