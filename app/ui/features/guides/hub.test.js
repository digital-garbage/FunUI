import test from "node:test";
import assert from "node:assert/strict";
import hub, { wanted } from "./index.js";

const clips = [{ sceneId: "c" }, { sceneId: "b" }, { sceneId: "a" }];          // nearest first

test("middle frame takes the middle of the clip before, held at the shot's middle", () => {
  assert.deepEqual(wanted("middle", 3, clips), [{ clip: clips[0], at: 0.5, middle: true }]);
});

test("memory takes each of the last clips' last pictures, oldest earliest, evenly spread", () => {
  const got = wanted("memory", 3, clips);
  assert.deepEqual(got.map((g) => g.clip.sceneId), ["a", "b", "c"]);
  assert.deepEqual(got.map((g) => g.at), [0.25, 0.5, 0.75]);
  assert.deepEqual(wanted("memory", 1, clips).map((g) => g.clip.sceneId), ["c"]);
  assert.deepEqual(wanted("off", 3, clips), []);
});

const media = { filename: "a.mp4", subfolder: "", type: "output" };
const slots = [{ id: "9", node: "FunUIGuideFrames" }];
const project = () => ({ id: "pj", num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a" }, { id: "b" }], scene_renders: { a: { media, inSec: 0 } } });
const make = (values, lastFrame) => { const app = { api: { lastFrame }, pipeline: { currentValues: () => ({ guide_frames: values }), allOff: () => false }, inputHooks: [] }; hub.setup({ app }); return app.inputHooks[0]; };

test("the frames reach the guides input as a list with the chosen strength", async () => {
  const asked = [];
  const hook = make({ mode: "middle", strength: 0.3 }, async (pid, body) => { asked.push(body); return { media_id: "m1" }; });
  const p = project();
  const out = await hook({ project: p, scene: p.scenes[1], slots });
  assert.deepEqual(JSON.parse(out.inputs[9].frames), [{ media_id: "m1", at: 0.5, strength: 0.3 }]);
  assert.equal(asked[0].at, 0.5);
});

test("nothing happens when off, without the input, for the first clip, or when a picture cannot be taken", async () => {
  const p = project();
  assert.deepEqual(await make({ mode: "off" }, async () => assert.fail("must not ask"))({ project: p, scene: p.scenes[1], slots }), {});
  assert.deepEqual(await make({ mode: "middle" }, async () => assert.fail("must not ask"))({ project: p, scene: p.scenes[1], slots: [] }), {});
  assert.deepEqual(await make({ mode: "middle" }, async () => assert.fail("must not ask"))({ project: p, scene: p.scenes[0], slots }), {});
  const broken = await make({ mode: "middle" }, async () => { throw new Error("no ffmpeg"); })({ project: p, scene: p.scenes[1], slots });
  assert.equal(broken.inputs, undefined);
  assert.match(broken.notes[0], /no ffmpeg/);
});
