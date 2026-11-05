// Continuity: a scene set to "From generated frame" starts from the last picture of the clip before it. This adds that picture to the
// scene's references for the run, after the ones it has, so it takes the next free reference slot (a generate input hook); the one choice
// lives in Project settings.
import { previousClip } from "../../shell/continuity.js";

export default {
  id: "continuity",
  mount: "menubar.menus",
  needs: ["project", "api", "pipeline", "inputHooks"],
  modules: ["render", "continuity"],
  setup({ app }) {
    const carryHook = async ({ project, scene, slots }) => {
      // Read from the pipeline this run was frozen with, not the live one an edit may have changed since the click.
      if (app.pipeline.allOff?.(slots)) return {};        // Disable all enhancements means a plain run
      const mine = (app.pipeline.currentValues(slots) || {}).continuity;
      if (!mine) return {};       // the module that holds the choices is not here: this does nothing at all
      const carry = mine.carry !== false;
      if (!carry || ((scene.source || {}).type || "carry") !== "carry") return {};      // a picture on the scene is the aspect source here (a pinned one is on every scene), not a start: it does not stop the carry
      const prev = previousClip(project, scene);
      if (!prev.sceneId) return prev.quiet ? {} : { notes: [`Starts without a picture: ${prev.why}.`] };
      let got;
      try { got = await app.api.lastFrame(project.id, { scene_id: prev.sceneId, render: prev.render, dur: prev.dur, src_in: prev.srcIn, reverse: prev.reverse }); }
      catch (err) { return { notes: [`Starts without a picture: the last frame of the clip before it could not be taken (${err.message}).`] }; }
      if (got.dark) return { notes: ["Starts without a picture: the clip before it ends in a fade to black."] };
      return { references: [got.media_id] };
    };
    const hook = async (ctx) => { const b = await carryHook(ctx); return { references: b.references || [], notes: b.notes || [] }; };
    app.inputHooks.push(hook);
    return () => { for (const [list, x] of [[app.inputHooks, hook]]) { const i = list.indexOf(x); if (i >= 0) list.splice(i, 1); } };
  },
};
