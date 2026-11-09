// Reference sheet: pick a grid, a picture and a label for each slot, and get one white sheet saved to the media bin.
// Pictures come from the bin (or are uploaded into it first); each is cut to the slot's shape on the server.
import { composer as c } from "../../composer/composer.js";
import { learn } from "../../shell/bin.js";
import { DEFAULTS, FORMATS, MAX_SLOTS, problem, request, resize } from "./sheet.js";

const thumb = (id) => `/fun/api/media/${encodeURIComponent(id)}/thumb`;

export default {
  id: "refsheet",
  mount: "assets.actions",
  needs: ["api", "dom"],
  modules: ["refsheet"],
  setup({ host, app }) {
    let win = null;
    const open = async () => {
      if (win) return;
      let pictures = [];
      const reload = async () => { const all = (await app.api.media()).media || []; learn(all); pictures = all.filter((m) => m.kind === "image"); };
      try { await reload(); } catch (err) { return c.toast.warn({ text: `Could not read the media bin: ${err.message}` }); }
      const state = { ...DEFAULTS, slots: resize([], DEFAULTS.rows, DEFAULTS.cols) };
      const grid = c.region.stack({ gap: "sm" });
      const note = c.text.sm({ text: "" });
      let make;

      const upload = async (i) => {
        const file = await app.dom.pickFile("image/*");
        if (!file) return;
        const got = await app.api.uploadMedia([file]);
        if (got.problems.length) c.toast.warn({ text: got.problems[0] });
        if (got.media[0]) { await reload(); state.slots[i].picture = got.media[0].id; app.say("bin"); }
      };
      // The bin itself, as pictures with their names: one click chooses, Escape or the close button keeps what was there.
      let picking = null;
      const choose = (i) => {
        if (picking) return;
        const done = (fn) => { picking.close("done"); Promise.resolve(fn && fn()).then(() => { drawGrid(); status(); }); };
        const items = pictures.map((m) => ({ id: m.id, label: m.name, thumb: thumb(m.id) }));
        const gallery = c.gallery.adaptive({ id: "refsheet-pick", items, selection: state.slots[i].picture ? [state.slots[i].picture] : [], empty: "The media bin has no pictures yet. Upload one.",
          onActivate: (it) => done(() => { state.slots[i].picture = it.id; }) });
        picking = c.modal.stacked({ title: `Picture for slot ${i + 1}`, size: "lg", onClose: () => { picking = null; }, body: gallery,
          actions: [c.button.md({ label: "Upload a picture…", onClick: () => done(() => upload(i)) }),
            c.button.md({ label: "Leave empty", tone: "ghost", onClick: () => done(() => { state.slots[i].picture = ""; }) })] });
      };
      const cell = (slot, i) => {
        const named = pictures.find((m) => m.id === slot.picture);
        const card = c.captioned.cell({ src: slot.picture ? thumb(slot.picture) : "", title: `Slot ${i + 1}`, badge: slot.picture && !named ? "picture gone" : "", tone: "warn", caption: slot.label, label: `Slot ${i + 1}`,
          onOpen: () => choose(i), onChange: (text) => { slot.label = text; } });
        const pick = c.button.sm({ label: named ? `Change: ${named.name}` : "Choose picture…", tone: "ghost", onClick: () => choose(i) });
        return c.region.stack({ gap: "xs", children: [card, pick] });
      };
      const drawGrid = () => grid.set([c.grid.cards({ children: state.slots.map(cell) })]);
      const status = () => { note.setText(problem(state)); make && make.setDisabled(!!problem(state)); };
      const regrid = () => { state.slots = resize(state.slots, state.rows, state.cols); drawGrid(); status(); };

      const num = (key, label, max) => c.field.default({ label, hint: `1 to ${max}`, control: c.number.md({ label, value: state[key], min: 1, max, step: 1, precision: 0, onChange: (v) => { state[key] = Math.max(1, Math.min(max, Math.round(v) || 1)); regrid(); } }) });
      make = c.button.md({ label: "Make sheet", tone: "primary", onClick: async () => {
        make.setBusy(true);
        try {
          const out = await app.api.refsheetMake(request(state));
          await reload(); app.say("bin");
          c.toast.good({ text: `Saved “${out.media.name}” to the media bin.` });
          if (out.missing.length) c.toast.warn({ text: `Left empty (picture gone or not readable): ${out.missing.join(", ")}.` });
          win.close("done");
        } catch (err) { c.toast.warn({ text: `Could not make the sheet: ${err.message}` }); } finally { make.setBusy(false); }
      } });
      win = c.modal.generic({ title: "Reference sheet", subtitle: "Pictures in a grid on white, each labelled below. Each picture is cropped from the middle to fit its slot.", size: "xl",
        onClose: () => { win = null; if (picking) picking.close("gone"); },
        body: c.region.stack({ gap: "md", children: [
          c.field.default({ label: "Title (top left corner, optional)", control: c.input.md({ value: state.title, label: "Title", onCommit: (v) => { state.title = v; } }) }),
          c.region.stack({ gap: "sm", children: [num("rows", "Rows", 12), num("cols", "Columns", 12),
            c.field.default({ label: "Slot format", hint: "width : height of every picture slot", control: c.select.md({ label: "Slot format", value: state.format, options: FORMATS.map((f) => ({ value: f, label: f })), onChange: (v) => { state.format = v; } }) })] }),
          note, grid] }),
        actions: [make] });
      drawGrid(); status();
    };

    const button = c.button.sm({ label: "Reference sheet…", tone: "ghost", title: `A labelled grid of pictures on white, saved to the media bin (up to ${MAX_SLOTS} slots)`, onClick: open });
    host.append(button.node);
    return () => { if (win) win.close("gone"); button.destroy(); };
  },
};
