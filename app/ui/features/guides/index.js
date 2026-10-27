// Guide frames: pictures from the clips before a shot go into its "Guide frames" input as weak guides (a generate input hook).
// The choices live in Settings > Engine > Guide frames; the work of laying them in is the pipeline's own node.
import { clipsBefore } from "../../shell/continuity.js";

const guideSlot = (slots) => (slots || []).find((s) => s.node === "FunUIGuideFrames");

/** -> [{ clip, at, middle }] the pictures wanted. `previous`: the clips before the shot, nearest first. Middle frame: the clip before, its middle picture,
 *  held at the shot's middle. Memory: each clip's last picture, the oldest earliest, spread evenly through the shot. */
export function wanted(mode, clips, previous) {
  if (mode === "middle") return previous.slice(0, 1).map((clip) => ({ clip, at: 0.5, middle: true }));
  if (mode !== "memory") return [];
  const list = previous.slice(0, Math.max(1, clips || 3)).reverse();
  return list.map((clip, i) => ({ clip, at: (i + 1) / (list.length + 1), middle: false }));
}

export default {
  id: "guides",
  mount: "menubar.menus",
  needs: ["project", "api", "pipeline", "inputHooks"],
  modules: ["render", "guide_frames"],
  setup({ app }) {
    const hook = async ({ project, scene, slots }) => {
      if (app.pipeline.allOff?.(slots)) return {};
      const mine = (app.pipeline.currentValues(slots) || {}).guide_frames;
      const slot = guideSlot(slots);
      if (!mine || !slot || !["middle", "memory"].includes(mine.mode) || project.generation_mode === "t2v") return {};
      const previous = clipsBefore(project, scene, mine.mode === "memory" ? mine.clips || 3 : 1);
      if (!previous.length) return {};
      const want = wanted(mine.mode, mine.clips, previous);
      const frames = [], notes = [];
      for (const { clip, at, middle } of want) {
        try {
          const got = await app.api.lastFrame(project.id, { scene_id: clip.sceneId, render: clip.render, dur: clip.dur, src_in: clip.srcIn, reverse: clip.reverse, ...(middle ? { at: 0.5 } : {}) });
          if (got.media_id) frames.push({ media_id: got.media_id, at, strength: mine.strength ?? 0.25 });
        } catch (err) { notes.push(`A guide picture could not be taken (${err.message}).`); }
      }
      if (!frames.length) return { notes };
      return { inputs: { [slot.id]: { frames: JSON.stringify(frames) } }, notes: [...notes, `Guide frames: ${frames.length} picture(s) from the clips before.`] };
    };
    app.inputHooks.push(hook);
    return () => { const i = app.inputHooks.indexOf(hook); if (i >= 0) app.inputHooks.splice(i, 1); };
  },
};
