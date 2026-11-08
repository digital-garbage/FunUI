// The guided tour: a walk along the screen, one callout per part, on whatever project is open (it changes nothing).
// Help ▸ Start tour / Restart tour start it, Exit tour ends it, Skip to FAQ jumps to the last callout.
import { composer as c } from "../../composer/composer.js";

// Each step points at a part of the screen by name: the menu bar, a zone, the page body, or a button by its label.
const STEPS = [
  { at: "menubar", side: "bottom", title: "Everything starts here", text: "The menu bar above has File, Edit, View and Help. Settings, next to Help, holds the models, the engine and the updates." },
  { at: "zone:0", side: "right", title: "Assets", text: "Your projects and the media bin. Drop images, clips and audio here; right-click a tile to use it as a reference, rename or delete it." },
  { at: "zone:1", side: "left", title: "Preview", text: "The monitor plays the cut wherever the playhead is. Save frame puts the current picture in the media bin." },
  { at: "zone:2", side: "left", title: "Properties", text: "Project settings (size, length, postfix) and the picked scene's prompt, source, references and length." },
  { at: "zone:3", side: "top", title: "Timeline", text: "One clip per scene. Drag the edges to trim, press S to split, Space to play. Lanes below carry separated sound." },
  { at: "button:▶ Generate|Generating…", side: "top", title: "Generate", text: "Makes every scene on the timeline, except left-out scenes and video clips; an earlier render of a scene stays as a take. Beside it: Selected makes only the picked scenes, Takes ×N makes the picked scene N times with only the seed changing, and Rate scene… lets FunUI learn your taste (Upscale and Interpolate appear there too once you switch them on). “Generate this scene” in Properties makes just the picked one." },
  { at: "button:⏱ Sampler", side: "top", title: "Sampler", text: "Steps, sampler, scheduler, seed and the common switches, one click from the timeline. While a run goes, the step count shows to its left." },
  { at: "button:◆ Composer", side: "top", title: "Composer", text: "Write the whole story in one box, cut at the cut word; keep templates, $variables and shortcuts; chat with the prompt enhancer." },
  { at: "button:＋ Add", side: "top", title: "Add and edit", text: "＋ Add opens one menu: a scene, a video clip or audio track from the bin, an effect or transition, a text or picture overlay. Next to it: Split, Remove, Export, Save to media bin and Separate audio." },
  { at: "button:⧉ Render", side: "top", title: "Render", text: "Joins the whole timeline into one file, with the separated sound mixed in. Export downloads the picked clips." },
  { at: "main", side: "bottom", title: "FAQ", text: "Nothing happens? Check the ComfyUI chip at the top right and the Log window. Wrong look? Settings ▸ Appearance. Lost a layout? View ▸ Reset Layout. Lost a project? File ▸ Open recent. That is the tour: Help ▸ Start tour runs it again." },
];

export default {
  id: "tour",
  mount: "menubar.menus",
  needs: ["project", "dom"],
  setup({ app }) {
    let at = -1, pop = null, target = null, watcher = null, queued = false;
    // The part pointed at can be redrawn (Generate becomes "Generating..."): the callout then follows the new node instead of a detached one.
    const follow = () => {
      if (queued || !pop || !target || target.isConnected) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; if (pop && at >= 0 && target && !target.isConnected) show(at); });
    };
    const stop = () => { if (watcher) { watcher.disconnect(); watcher = null; } target = null; if (pop) { const p = pop; pop = null; p.close(); } at = -1; app.has.delete("tour.running"); };      // a flag the Help menu reads: features see copies of app, the Set is shared
    const shown = (i) => { const el = app.dom.landmark(STEPS[i].at); return el && el.getBoundingClientRect().width ? el : null; };
    function show(i) {
      if (pop) { const p = pop; pop = null; p.close(); }
      target = null;
      while (i >= 0 && i < STEPS.length && !(target = shown(i))) i += i >= at ? 1 : -1;      // a part that is hidden right now is skipped
      if (i < 0 || i >= STEPS.length) return stop();
      at = i;
      app.has.add("tour.running");
      if (!watcher && typeof MutationObserver === "function") { watcher = new MutationObserver(follow); watcher.observe(app.dom.body, { childList: true, subtree: true }); }
      const seen = STEPS.map((_, k) => k).filter((k) => k === i || shown(k)), place = seen.indexOf(i) + 1;       // numbered among the parts on screen, so the count has no gaps
      const s = STEPS[i], last = i === STEPS.length - 1;
      const body = c.region.stack({ gap: "sm", children: [c.label.section({ text: `${place} / ${seen.length} · ${s.title}` }), c.text.sm({ text: s.text }),
        c.toolbar.default({ items: [c.button.sm({ label: "Back", tone: "ghost", disabled: i === 0, onClick: () => show(i - 1) }), c.button.sm({ label: last ? "Done" : "Next", tone: "primary", onClick: () => (last ? stop() : show(i + 1)) })],
          trailing: [c.button.sm({ label: "Exit tour", tone: "ghost", onClick: stop })] })] });
      pop = c.popover.anchored({ anchor: target, body, side: s.side, align: "center", gap: 10, closeOnOutside: false, onClose: () => { if (pop) { pop = null; at = -1; if (watcher) { watcher.disconnect(); watcher = null; } app.has.delete("tour.running"); } } });
    }
    app.has.add("tour");
    const off = app.on((what) => {
      if (what === "tour.start") show(0);
      else if (what === "tour.faq") show(STEPS.length - 1);
      else if (what === "tour.stop") stop();
    });
    return () => { off(); app.has.delete("tour"); app.has.delete("tour.running"); };
  },
};
