// Interpolate a finished render with NVIDIA RTX Frame Generation: more frames, same length and sound; the result replaces the render. Never / button beside the rating / always.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { derived, keyOf, swap, wentThrough } from "../../shell/renders.js";
import { DEFAULTS, RTX, interpolate, modeLabel } from "./job.js";

export default {
  id: "interpolate",
  mount: "timeline.actions",
  needs: ["project", "api", "selection"],
  modules: ["interpolate"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host), busy = new Set();
    const render = (id) => ((p.project || {}).scene_renders || {})[id];
    const target = () => {                               // the picked clip's generation unit root, when it has a render
      const open = p.project, sc = open && open.scenes.find((s) => s.id === sel.focus);
      const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
      const r = root && render(root.id);
      return r && r.media ? r.media : null;
    };
    const opts = () => ({ ...DEFAULTS, ...(p.pref("interpolateOpts", {}) || {}) });
    // Pressed with nothing to interpolate with: say what would fix it, and take the person there.
    async function suggest() {
      let all = []; try { all = await app.api.addons(); } catch { /* no add-on list */ }
      const fg = all.find((a) => a.id === "nvidia_framegen");
      if (!fg) return void c.toast.warn({ text: `${RTX} needs an NVIDIA RTX card on Windows or Linux, so it is not available here.` });
      const pick = await c.modal.choice({ title: "Nothing to interpolate with yet", items: [fg.state === "pending"
        ? { id: "modules", label: `${RTX} is switched on`, hint: "Restart ComfyUI to install it." }
        : { id: "modules", label: `Install ${RTX}`, hint: "Switch it on in Settings ▸ Modules. It needs nvidia-vfx 0.2 or newer and moves an older one up." }] }).result;
      if (pick) app.openSettings(pick);
    }

    async function run(media, pressed = false) {
      if (!media || wentThrough(media, "interpolate")) return;                // never interpolate twice
      const key = keyOf(media), pid = p.project.id;
      if (busy.has(key)) return;
      busy.add(key); draw();
      try {
        // Only counts when this machine has it on right now: a project saved elsewhere may still say "always"
        if (!(await app.api.frameGeneration().then((r) => Boolean(r.rtx)).catch(() => false)))
          return void (pressed ? suggest() : c.toast.warn({ text: `${RTX} is not available on this machine (or is switched off in Settings ▸ Modules).` }));
        c.toast.info({ text: `Interpolating with ${RTX}…` });
        const out = await interpolate(app.api, media, opts());
        derived(media, out, "interpolate");
        const swapped = await p.editFor(pid, (pr) => swap(pr, media, out) > 0);
        if (!swapped) c.toast.warn({ text: "Interpolated, but that render was replaced meanwhile, so nothing was swapped." });
        else c.toast.good({ text: `Interpolated: the render now plays at ${opts().multiplier} times the frames.` });
      } catch (err) { c.toast.warn({ text: `Interpolate failed: ${err.message}` }); }
      finally { busy.delete(key); draw(); }
    }

    function draw() {
      const media = p.pref("interpolateMode", "never") === "button" ? target() : null;
      if (!media) return put();
      const done = wentThrough(media, "interpolate"), working = busy.has(keyOf(media));
      put(c.button.sm({ label: working ? "Interpolating…" : done ? "Interpolated" : "Interpolate", tone: "ghost", disabled: working || done,
        title: "Add in-between frames to this video with NVIDIA RTX Frame Generation; it replaces the render", onClick: () => run(media, true) }).node);
    }

    // "Always": a render that appears after the project was opened is interpolated. What was there on open is left alone.
    let seen = null, obj = null;
    const watch = () => {
      const open = p.project;
      if (!open) { seen = null; obj = null; return; }
      const now = new Map(Object.entries(open.scene_renders || {}).filter(([, r]) => r && r.media).map(([id, r]) => [id, keyOf(r.media)]));
      if (open !== obj || !seen) { obj = open; seen = now; return; }       // another project, or undo/redo put an older copy back: not a new render
      if (p.pref("interpolateMode", "never") === "always") {
        const fresh = new Map([...now].filter(([id, k]) => seen.get(id) !== k));
        const pending = (m) => p.pref("upscaleMode", "never") === "always" && !wentThrough(m, "upscale");   // Upscale goes first; its result comes back as a new render
        for (const media of new Map([...fresh.keys()].map((id) => [fresh.get(id), open.scene_renders[id].media])).values()) if (!pending(media)) run(media);
      }
      seen = now;
    };

    app.editorSettings.push(() => {
      const page = c.region.stack({ gap: "md" });
      const set = (k, v) => { p.setPref("interpolateOpts", { ...opts(), [k]: v }); draw(); };
      const paint = (r) => {
        const o = opts();
        page.set([
          c.label.section({ text: "Interpolate finished renders" }),
          c.hint.default({ text: "Adds in-between frames to a finished video and replaces the render with the result. The length and the sound stay, the frame rate goes up." }),
          c.settingsRow.default({ label: "Interpolate", hint: "Never, with a button beside the rating, or on every new render.", control: c.select.md({ label: "Interpolate", value: p.pref("interpolateMode", "never"),
            options: [{ value: "never", label: "Never" }, { value: "button", label: "When I press Interpolate" }, { value: "always", label: "Always, when a render finishes" }], onChange: (v) => { p.setPref("interpolateMode", v); draw(); } }) }),
          ...(r && r.rtx ? [
            c.settingsRow.default({ label: "Frames", hint: "2 times turns 24 fps into 48 fps.", control: c.select.md({ label: "Frames", value: String(o.multiplier), onChange: (v) => set("multiplier", Number(v)),
              options: Array.from({ length: r.rtx.max_multiplier - 1 }, (_, i) => ({ value: String(i + 2), label: `${i + 2} times` })) }) }),
            c.settingsRow.default({ label: "Quality", hint: "Fast is for a quick look, Best for the final video.", control: c.select.md({ label: "Quality", value: o.mode, onChange: (v) => set("mode", v),
              options: r.rtx.modes.map((m) => ({ value: m, label: modeLabel(m) })) }) }),
            c.toggle.default({ label: "10-bit processing", hint: "More precise colour inside the interpolation. The saved video stays 8-bit.", checked: o.tenbit, onChange: (v) => set("tenbit", v) }),
          ] : r === null ? [] : [c.banner.warn({ text: !r.rtx_offered ? `${RTX} needs an NVIDIA RTX card on Windows or Linux, so it is not available here.` : `${RTX} is not installed or is switched off: switch it on in Settings ▸ Modules and restart ComfyUI.` })]),
        ]);
      };
      paint(null);
      app.api.frameGeneration().then(paint).catch(() => page.set([c.banner.warn({ text: "Could not read the frame generation setup." })]));
      return page;
    });

    draw(); watch();
    return app.on(() => { watch(); draw(); });
  },
};
