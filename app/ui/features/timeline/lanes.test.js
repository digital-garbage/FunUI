import test from "node:test";
import assert from "node:assert/strict";
import { videoLane } from "./lanes.js";

test("scenes, a ghost and a pause become clips laid end to end", () => {
  const p = { num_frames_per_scene: 97, frame_rate: 25, scene_ghosts: [{ id: "g", afterSceneId: "a", text: "gone", durationSec: 2 }],
    scenes: [{ id: "a", text: "one\ntwo", gap_after_sec: 1 }, { id: "b", text: "" }] };
  const lane = videoLane(p, ["b"], "b");
  assert.deepEqual(lane.clips.map((c) => c.id), ["a", "ghost:g", "gap:a", "b"]);
  assert.equal(lane.clips[0].title, "one");
  assert.equal(lane.clips[3].selected, true);
  assert.equal(lane.clips[3].start, lane.clips[2].start + 1);
});

test("a rated clip says so: liked or disliked, whichever word the rating is saved in", async () => {
  const { mood } = await import("./lanes.js");
  assert.deepEqual(["10", "3", "Disliked: bad image", "7|loved", "", "-Just forget it-", "weird", "Perfect", "Awful", "Wrong appearance"].map(mood), ["liked", "disliked", "disliked", "liked", "", "", "", "liked", "disliked", ""]);
});

test("a picture render shows as its picture on the clip; a video render keeps the picture the scene starts from", () => {
  const p = { num_frames_per_scene: 1, frame_rate: 1,
    scenes: [{ id: "a", text: "x", source_image: "m1" }, { id: "b", text: "y", source_image: "m2" }],
    scene_renders: { a: { media: { filename: "FunUI_00001_.png", subfolder: "s", type: "output" } }, b: { media: { filename: "b.mp4", subfolder: "", type: "output" } } } };
  const [a, b] = videoLane(p, [], "").clips;
  const u = new URL(a.thumbs[0], "http://x");
  assert.equal(u.pathname, "/fun/api/m/render/poster");
  assert.deepEqual(Object.fromEntries(u.searchParams), { filename: "FunUI_00001_.png", subfolder: "s", type: "output" });
  assert.deepEqual(b.thumbs, ["/fun/api/media/m2/thumb"]);
});
