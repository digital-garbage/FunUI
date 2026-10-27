// Which clip a shot continues from: the one right before its generation unit on the timeline.
import { segments, genUnitId, isVideoClip } from "./scenes.js";

/** -> { sceneId, render, dur, srcIn } for the clip a shot should continue from, or { why } when there is nothing to continue from. */
export function previousClip(p, scene) {
  const unit = genUnitId(scene), segs = segments(p).filter((s) => s.kind === "scene");
  const first = segs.findIndex((s) => genUnitId(s.scene) === unit);
  if (first <= 0) return { why: "it is the first clip", quiet: true };
  const prev = segs[first - 1];
  if (prev.scene.excluded) return { why: "the clip before it is left out" };
  if (isVideoClip(prev.scene)) return { why: "the clip before it is an imported video" };
  const render = (p.scene_renders || {})[prev.scene.id];
  if (!(render && render.media)) return { why: "the clip before it has no render yet" };
  return { sceneId: prev.scene.id, render: { media: render.media, inSec: render.inSec || 0 }, dur: prev.dur, srcIn: prev.scene.source_in || 0, reverse: Boolean((prev.scene.effects || {}).reverse) };       // a reversed clip ends on its render's first picture
}

/** -> up to `n` of the clips before a shot's generation unit, nearest first, each { sceneId, render, dur, srcIn, reverse }; clips left out, imported or not yet rendered are skipped. */
export function clipsBefore(p, scene, n) {
  const unit = genUnitId(scene), segs = segments(p).filter((s) => s.kind === "scene");
  const first = segs.findIndex((s) => genUnitId(s.scene) === unit);
  const out = [];
  for (let i = first - 1; i >= 0 && out.length < n; i--) {
    const prev = segs[i], render = (p.scene_renders || {})[prev.scene.id];
    if (prev.scene.excluded || isVideoClip(prev.scene) || !(render && render.media)) continue;
    out.push({ sceneId: prev.scene.id, render: { media: render.media, inSec: render.inSec || 0 }, dur: prev.dur, srcIn: prev.scene.source_in || 0, reverse: Boolean((prev.scene.effects || {}).reverse) });
  }
  return out;
}
