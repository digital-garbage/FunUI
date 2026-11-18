// Continuity: a scene set to "From generated frame" starts from the last picture of the clip before it. This adds that picture to the
// scene's references for the run, after the ones it has, so it takes the next free reference slot (a generate input hook). It can also
// pass the clip's last N frames as a video (with its own sound) and/or its sound alone; the choices live in Project settings.
import { previousClip } from "../../shell/continuity.js";
import { kinds, sounds } from "../../shell/bin.js";

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
      if (((scene.source || {}).type || "carry") !== "carry") return {};      // a picture on the scene is the aspect source here (a pinned one is on every scene), not a start: it does not stop the carry
      const frame = mine.carry !== false, video = mine.tail_video === true, audio = mine.tail_audio === true;
      if (!frame && !video && !audio) return {};
      const prev = previousClip(project, scene);
      if (!prev.sceneId) return prev.quiet ? {} : { notes: [`Starts without ${frame ? "a picture" : "the end of the clip before"}: ${prev.why}.`] };
      const window = { scene_id: prev.sceneId, render: prev.render, dur: prev.dur, src_in: prev.srcIn, reverse: prev.reverse };
      const references = [], notes = [];
      if (frame) {
        try {
          const got = await app.api.lastFrame(project.id, window);
          if (got.dark) notes.push("Starts without a picture: the clip before it ends in a fade to black.");
          else references.push(got.media_id);
        } catch (err) { notes.push(`Starts without a picture: the last frame of the clip before it could not be taken (${err.message}).`); }
      }
      if (video || audio) {
        try {
          const got = await app.api.tail(project.id, { ...window, frames: Math.max(1, Math.round(Number(mine.tail_frames) || 50)), video, audio });
          if (got.video_id) { kinds.set(got.video_id, "video"); if (got.sound) sounds.add(got.video_id); references.push(got.video_id); }
          if (got.audio_id) { kinds.set(got.audio_id, "audio"); references.push(got.audio_id); }
          notes.push(...(got.notes || []));
        } catch (err) { notes.push(`The end of the clip before it could not be carried (${err.message}).`); }
      }
      return { references, notes };
    };
    const hook = async (ctx) => { const b = await carryHook(ctx); return { references: b.references || [], notes: b.notes || [] }; };
    app.inputHooks.push(hook);
    return () => { for (const [list, x] of [[app.inputHooks, hook]]) { const i = list.indexOf(x); if (i >= 0) list.splice(i, 1); } };
  },
};
