import test from "node:test";
import assert from "node:assert/strict";
import hub from "./index.js";

const media = { filename: "a.mp4", subfolder: "", type: "output" };
const slots = [{ id: "7", roles: [{ at: "assets.reference_1", input: "media_id" }] }];
const project = (extra = {}) => ({ id: "pj", num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a" }, { id: "b", source: { type: "carry" } }], scene_renders: { a: { media, inSec: 0 } }, ...extra });
const make = (values, lastFrame) => { const app = { api: { lastFrame }, pipeline: { currentValues: () => ({ continuity: values }) }, inputHooks: [], mediaMenu: [], project: { project: null, edit: () => true } }; hub.setup({ app }); return app.inputHooks[0]; };

test("a carry shot gets the previous clip's last frame as an extra reference, also when a picture (the aspect source) is set on it", async () => {
  const calls = [];
  const hook = make({ carry: true }, async (pid, body) => { calls.push([pid, body.scene_id, body.dur]); return { media_id: "m9", dark: false }; });
  const p = project();
  p.scenes[1].source_image = "aspect";
  assert.deepEqual((await hook({ project: p, scene: p.scenes[1], slots })).references, ["m9"]);
  assert.deepEqual([calls[0][0], calls[0][1], Math.round(calls[0][2] * 25)], ["pj", "a", 97]);
});

test("nothing happens when it is off or the source is another kind", async () => {
  const hook = make({ carry: false }, async () => assert.fail("must not ask"));
  const p = project();
  assert.deepEqual(await hook({ project: p, scene: p.scenes[1], slots }), { references: [], notes: [] });
  const on = make({}, async () => assert.fail("must not ask"));
  p.scenes[1].source = { type: "image" };
  assert.deepEqual(await on({ project: p, scene: p.scenes[1], slots }), { references: [], notes: [] });
});

test("a dark last frame, a failed extraction and a missing predecessor each say why the shot starts without a picture", async () => {
  const p = project();
  const dark = make({}, async () => ({ media_id: "m", dark: true }));
  assert.match((await dark({ project: p, scene: p.scenes[1], slots })).notes[0], /fade to black/);
  const broken = make({}, async () => { throw new Error("no ffmpeg"); });
  assert.match((await broken({ project: p, scene: p.scenes[1], slots })).notes[0], /no ffmpeg/);
  assert.deepEqual((await broken({ project: p, scene: p.scenes[0], slots })).notes || [], []);       // the first clip has nothing to continue from: silent
});

const make2 = (values, api) => { const app = { api, pipeline: { currentValues: () => ({ continuity: values }) }, inputHooks: [], mediaMenu: [], project: { project: null, edit: () => true } }; hub.setup({ app }); return app.inputHooks[0]; };

test("the end of the clip before goes on as a video with its sound and/or a sound, sized in frames, as the right kinds", async () => {
  const { kinds, sounds } = await import("../../shell/bin.js");
  const asked = [];
  const tail = async (pid, body) => { asked.push(body); return { video_id: "v1", audio_id: "a1", sound: true, notes: ["The file as generated is gone."] }; };
  const hook = make2({ carry: false, tail_video: true, tail_audio: true, tail_frames: 37 }, { tail, lastFrame: async () => assert.fail("the last picture is off") });
  const p = project();
  const out = await hook({ project: p, scene: p.scenes[1], slots });
  assert.deepEqual(out.references, ["v1", "a1"]);
  assert.deepEqual([asked[0].frames, asked[0].video, asked[0].audio, asked[0].scene_id], [37, true, true, "a"]);
  assert.deepEqual([kinds.get("v1"), kinds.get("a1"), sounds.has("v1")], ["video", "audio", true]);
  assert.match(out.notes[0], /as generated/);
});

test("sound alone asks for no video; a failed cut says so and the picture still goes", async () => {
  const asked = [];
  const hook = make2({ tail_audio: true }, { tail: async (pid, body) => { asked.push(body); throw new Error("no ffmpeg"); }, lastFrame: async () => ({ media_id: "m9", dark: false }) });
  const p = project();
  const out = await hook({ project: p, scene: p.scenes[1], slots });
  assert.deepEqual([asked[0].video, asked[0].audio], [false, true]);
  assert.deepEqual(out.references, ["m9"]);
  assert.match(out.notes[0], /no ffmpeg/);
});

test("with nothing ticked and the picture off nothing is asked", async () => {
  const hook = make2({ carry: false }, { tail: async () => assert.fail("no"), lastFrame: async () => assert.fail("no") });
  const p = project();
  assert.deepEqual(await hook({ project: p, scene: p.scenes[1], slots }), { references: [], notes: [] });
});
