import test from "node:test";
import assert from "node:assert/strict";
import { buildInputs } from "./inputs.js";
import { kinds } from "../../shell/bin.js";

const slots = [
  { id: "p", roles: [{ at: "generation.prompt", input: "text" }] },
  { id: "s", roles: [{ at: "assets.source_image" }] },
  { id: "k", roles: [{ at: "project.video", input: "steps" }, { at: "project.negative", input: "neg" }] },
];

test("each value lands on the slot whose role asks for it", async () => {
  const { inputs } = await buildInputs({ project: { video: { steps: 8 }, negative: "blur" }, scene: { text: "a cat", source_image: "m1" }, slots,
    expand: async ({ text }) => ({ text: `[${text}]` }) });
  assert.deepEqual(inputs, { p: { text: "[a cat]" }, s: { media_id: "m1" }, k: { steps: 8, neg: "blur" } });
});

test("a failing expander sends the typed text", async () => {
  const { inputs } = await buildInputs({ project: {}, scene: { text: "x" }, slots, expand: async () => { throw new Error("down"); } });
  assert.equal(inputs.p.text, "x");
});

test("a failing expander is said, not silent", async () => {
  const r = await buildInputs({ project: {}, scene: { text: "x" }, slots, expand: async () => null });
  assert.ok(r.notes.some((n) => /could not be expanded/.test(n)));
  const ok = await buildInputs({ project: {}, scene: { text: "x" }, slots, expand: async () => ({ text: "x" }) });
  assert.ok(!ok.notes.some((n) => /could not be expanded/.test(n)));
});

test("a role that drives length is fed from the project number the timeline draws", async () => {
  const s = [{ id: "k", roles: [{ at: "project.video", input: "length", drives: "frames" }, { at: "project.video", input: "width" }] }];
  const { inputs } = await buildInputs({ project: { num_frames_per_scene: 49, video: { length: 97, width: 640 } }, scene: {}, slots: s, expand: async () => null });
  assert.deepEqual(inputs.k, { length: 49, width: 640 });
});

test("no prompt role is reported, not silently ignored", async () => {
  const { noPrompt } = await buildInputs({ project: {}, scene: { text: "x" }, slots: [], expand: async () => null });
  assert.equal(noPrompt, true);
});

test("each build draws a fresh seed, and a unit's own length overrides the project's", async () => {
  const s = [{ id: "n", roles: [{ at: "generation.seed", input: "seed" }, { at: "project.video", input: "length", drives: "frames" }] }];
  const args = { project: { num_frames_per_scene: 97 }, scene: {}, slots: s, expand: async () => null, frames: 49 };
  const a = await buildInputs({ ...args, seed: () => 1 }), b = await buildInputs({ ...args, seed: () => 2 });
  assert.deepEqual([a.inputs.n.seed, b.inputs.n.seed, a.inputs.n.length], [1, 2, 49]);
});

test("starting shots from a prompt sends no picture even when the scene has one", async () => {
  const { inputs } = await buildInputs({ project: { generation_mode: "t2v" }, scene: { text: "a cat", source_image: "m1" }, slots, expand: async ({ text }) => ({ text }) });
  assert.equal(inputs.s.media_id, "");
});

test("a hook's inputs are merged and its notes come back", async () => {
  const hook = ({ scene }) => ({ inputs: { k: { extra: scene.text } }, notes: ["said"] });
  const r = await buildInputs({ project: {}, scene: { text: "x" }, slots, hooks: [hook, () => null], expand: async ({ text }) => ({ text }) });
  assert.equal(r.inputs.k.extra, "x");
  assert.deepEqual(r.notes, ["said"]);
});

test("a hook that throws does not stop the run: its problem comes back as a note", async () => {
  const r = await buildInputs({ project: {}, scene: { text: "x" }, slots, hooks: [() => { throw new Error("boom"); }], expand: async ({ text }) => ({ text }) });
  assert.equal(r.inputs.p.text, "x");
  assert.match(r.notes[0], /boom/);
});

test("a shot with no text and no start picture is said; text from the project's anchor counts as text", async () => {
  const expand = async ({ text, anchor }) => ({ text: [anchor, text].filter(Boolean).join(" ") });
  const said = async (project, scene) => (await buildInputs({ project, scene, slots, expand })).notes.some((n) => /no text/.test(n));
  assert.equal(await said({}, { text: "  " }), true);
  assert.equal(await said({ anchor: "a cat" }, { text: "" }), false);
  assert.equal(await said({}, { text: "", source_image: "m1" }), false);
});

test("the no-text note goes by what is sent: a carried frame counts, a picture t2v blanks does not", async () => {
  const expand = async ({ text }) => ({ text });
  const carry = [async () => ({ inputs: { s: { media_id: "frame" } } })];
  const noted = async (project, scene, hooks = []) => (await buildInputs({ project, scene, slots, expand, hooks })).notes.some((n) => /no text/.test(n));
  assert.equal(await noted({}, { text: "" }, carry), false);
  assert.equal(await noted({ generation_mode: "t2v" }, { text: "", source_image: "m1" }), true);
});

test("a fixed seed (seed: null) sends no seed: the seed node's own number runs", async () => {
  const s = [{ id: "n", roles: [{ at: "generation.seed", input: "seed" }], inputs: { seed: 7 } }];
  const { inputs } = await buildInputs({ project: {}, scene: {}, slots: s, expand: async () => null, seed: null });
  assert.equal((inputs.n || {}).seed, undefined);
});

test("the shot camera node is told which project this is, whatever roles the saved pipeline carries", async () => {
  const slots = [{ id: "shot_camera", node: "FunUIShotCamera", roles: [] }];
  const { inputs } = await buildInputs({ project: { id: "proj1", scenes: [], video: {} }, scene: { id: "s", text: "x", references: [] }, slots, expand: async ({ text }) => ({ text }) });
  assert.equal(inputs.shot_camera.project, "proj1");
});

test("text-only: a slot that can stand in for the picture is told the size of the slot that shapes it, and a picked picture is sent as is", async () => {
  const like = { slot: "t", width: "target_width", height: "target_height", unsend: { slot: "e", input: "use_image" } };
  const s = [{ id: "src", roles: [{ at: "assets.source_image", input: "media_id", blank_like: like }] }, { id: "e", inputs: { use_image: true } },
    { id: "t", inputs: { target_width: 1344, target_height: 768 }, roles: [{ at: "project.video", input: "target_width" }] }];
  const run = (project, scene) => buildInputs({ project, scene, slots: s, expand: async () => null });
  const t2v = await run({ generation_mode: "t2v", video: { target_width: 640 } }, { source_image: "m1" });
  assert.deepEqual(t2v.inputs.src, { media_id: "", blank_width: 640, blank_height: 768 });
  assert.deepEqual(t2v.inputs.e, { use_image: false });                          // the enhancer is not handed the grey stand-in
  assert.equal((await run({ generation_mode: "i2v" }, { source_image: "m1" })).inputs.e, undefined);
  assert.deepEqual((await run({ generation_mode: "i2v" }, { source_image: "m1" })).inputs.src, { media_id: "m1" });
  assert.deepEqual((await run({ generation_mode: "i2v" }, {})).inputs.src, { media_id: "", blank_width: 1344, blank_height: 768 });
});

test("text-only on a pipeline saved before the preset named a stand-in: the resizer slot is enough, no preset change needed", async () => {
  const s = [{ id: "src", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id" }] }, { id: "e", node: "FunUIEnhancePrompt", inputs: { use_image: true } },
    { id: "t", node: "ImageTransformKJ", inputs: { target_width: 1344, target_height: 768 }, roles: [{ at: "project.video", input: "target_width" }] }];
  const { inputs } = await buildInputs({ project: { generation_mode: "t2v", video: { target_width: 640 } }, scene: {}, slots: s, expand: async () => null });
  assert.deepEqual(inputs.src, { media_id: "", blank_width: 640, blank_height: 768 });
  assert.deepEqual(inputs.e, { use_image: false });
});

test("a video reference reaches the model's video input from a preset that says nothing about videos", async () => {
  const { learn } = await import("../../shell/bin.js");
  learn([{ id: "vid", name: "v.mp4", kind: "video" }]);
  const s = [{ id: "ref1", node: "Loader", roles: [{ at: "assets.reference_1", input: "media_id", wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } }] },
    { id: "r2v", node: "Model", inputs: {} }];
  const info = { Model: { inputs: { "ref_images.ref_image_0": "IMAGE", "ref_videos.ref_video_0": "IMAGE" }, outputs: [] }, Loader: { inputs: {}, outputs: ["IMAGE", "IMAGE", "AUDIO"] } };
  const r = await buildInputs({ project: { video: {} }, scene: { text: "x", references: ["vid"] }, slots: s, expand: async () => null, nodeInfo: async (c) => info[c] });
  assert.equal(r.unwired, 0);
  assert.deepEqual(r.inputs.r2v["ref_videos.ref_video_0"], ["ref1", 1]);
  assert.equal(r.inputs.ref1.full, true);
  learn([]);
});

test("a reference video is prepared as the project says, else as the scene says, and untouched when nothing is on", async () => {
  const { learn } = await import("../../shell/bin.js");
  learn([{ id: "vid", name: "v.mp4", kind: "video" }]);
  const s = [{ id: "ref1", node: "Loader", roles: [{ at: "assets.reference_1", input: "media_id", wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } }] }, { id: "r2v", node: "Model", inputs: {} }];
  const info = { Model: { inputs: { "ref_videos.ref_video_0": "IMAGE" }, outputs: [] }, Loader: { inputs: {}, outputs: ["IMAGE", "IMAGE", "AUDIO"] } };
  const run = async (project, scene) => (await buildInputs({ project: { video: {}, frame_rate: 24, ...project }, scene: { text: "x", references: ["vid"], ...scene }, slots: s, expand: async () => null, nodeInfo: async (c) => info[c] })).inputs.ref1;
  assert.equal((await run({}, {})).prepare, undefined);
  const scene = { video_input: { on: true, skip_first: 5, every_nth: 2 } };
  const sc = JSON.parse((await run({}, scene)).prepare);
  assert.equal(sc.skip_first, 5);
  assert.equal(sc.every_nth, 2);
  assert.equal(sc.fps, 24);          // "the project's FPS"
  const both = { video_input: { on: true, skip_first: 9, fps_from: "source" } };
  assert.equal(JSON.parse((await run(both, scene)).prepare).skip_first, 9);          // auto: the project's wins when on
  assert.equal(JSON.parse((await run(both, scene)).prepare).fps, 0);
  assert.equal(JSON.parse((await run({ ...both, video_input_from: "scene" }, scene)).prepare).skip_first, 5);
  assert.equal((await run({ video_input_from: "project", video_input: { on: false } }, scene)).prepare, undefined);
  learn([]);
});

test("a video picked as the start feeds its frames to what the picture fed, when the pipeline says it can", async () => {
  const { learn } = await import("../../shell/bin.js");
  learn([{ id: "vid", name: "v.mp4", kind: "video" }, { id: "pic", name: "p.png", kind: "image" }]);
  const mk = (flag) => [{ id: "src", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id", ...(flag ? { video_frames: true } : {}) }], inputs: {} },
    { id: "anchor", node: "X", inputs: { image: ["src", 0], other: ["src", 1] } }];
  const run = (slots, id, project = {}) => buildInputs({ project: { video: {}, frame_rate: 25, ...project }, scene: { text: "x", source_image: id }, slots, expand: async () => null });
  const on = await run(mk(true), "vid", { video_input: { on: true, skip_first: 3 } });
  assert.equal(on.inputs.src.full, true);
  assert.equal(JSON.parse(on.inputs.src.prepare).skip_first, 3);
  assert.equal((await run(mk(true), "vid")).inputs.src.prepare, undefined);          // no preparation switched on: whole video as before
  assert.equal((await run(mk(false), "vid")).inputs.src.full, undefined);          // the pipeline did not say: first frame only
  assert.equal((await run(mk(true), "pic")).inputs.src.full, undefined);          // a picture stays a picture
  learn([]);
});

test("a video-in pipeline takes the first video among the references as its start, and says so when there is none", async () => {
  const { learn } = await import("../../shell/bin.js");
  learn([{ id: "vid", name: "v.mp4", kind: "video" }, { id: "pic", name: "p.png", kind: "image" }]);
  const slots = [{ id: "src", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id", video_frames: true, video_only: true }], inputs: {} }, { id: "a", node: "X", inputs: { image: ["src", 0] } }];
  const run = (scene, project = {}) => buildInputs({ project: { video: {}, frame_rate: 25, ...project }, scene: { text: "x", ...scene }, slots, expand: async () => null });
  const r = await run({ references: ["pic", "vid"] }, { generation_mode: "t2v" });
  assert.equal(r.inputs.src.media_id, "vid");
  assert.equal(r.inputs.src.full, true);
  assert.equal((await run({ source_image: "pic", references: ["vid"] })).inputs.src.media_id, "pic");          // a picked start wins
  assert.ok((await run({ references: ["pic"] })).notes.some((n) => /works from a video/.test(n)));
  learn([]);
});

test("an LTX image-to-video pipeline starts from the first reference picture when none is picked, and keeps it as a reference", async () => {
  const { learn } = await import("../../shell/bin.js");
  learn([{ id: "vid", name: "v.mp4", kind: "video" }, { id: "pic", name: "p.png", kind: "image" }, { id: "pic2", name: "q.png", kind: "image" }]);
  const slots = [{ id: "src", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id", video_frames: true }], inputs: {} }];
  const run = (scene, project = {}) => buildInputs({ project: { video: {}, frame_rate: 25, ...project }, scene: { text: "x", ...scene }, slots, expand: async () => null });
  assert.equal((await run({ references: ["pic", "pic2"] })).inputs.src.media_id, "pic");
  assert.equal((await run({ references: ["pic", "vid"] })).inputs.src.media_id, "vid");          // a video is the start when there is one
  assert.equal((await run({ source_image: "pic2", references: ["pic"] })).inputs.src.media_id, "pic2");          // a picked start wins
  assert.equal((await run({ references: ["pic"] }, { generation_mode: "t2v" })).inputs.src.media_id, "");          // "From a prompt": no picture
  learn([]);
});

test("a direct video-to-video pipeline saved without the video-only flag is still video-to-video (an encoder is fed by the start slot)", async () => {
  const { learn } = await import("../../shell/bin.js");
  learn([{ id: "vid", name: "v.mp4", kind: "video" }, { id: "pic", name: "p.png", kind: "image" }]);
  const slots = [{ id: "src", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id", video_frames: true }], inputs: {} },
    { id: "enc", node: "VAEEncode", inputs: { pixels: ["src", 1] } }];
  const r = await buildInputs({ project: { video: {}, generation_mode: "t2v" }, scene: { text: "x", references: ["pic", "vid"] }, slots, expand: async () => null });
  assert.equal(r.inputs.src.media_id, "vid");
  assert.equal(r.inputs.src.full, true);
  learn([]);
});

test("a reference whose target node was deleted from an edited pipeline is left unused, not put into a slot that is not there", async () => {
  const slots = [{ id: "identity_source", node: "FunUILoadMedia", roles: [{ at: "assets.reference_1", input: "media_id", wireTo: { slot: "identity", input: "image" } }], inputs: { media_id: "" } }];
  kinds.set("pic", "image");
  const r = await buildInputs({ project: { id: "P", video: {} }, scene: { id: "s", text: "", references: ["pic"] }, slots, expand: async (x) => ({ text: x.text }), nodeInfo: async () => null, frames: 49 });
  assert.equal(r.inputs.identity, undefined);
  assert.equal(r.unwired, 1);
  assert.match(r.left[0].why, /identity/);
  assert.equal(r.left[0].id, "pic");
});

const LTX = {
  refs: [
    { id: "aspect_source", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id", blank_like: { slot: "aspect", width: "width", height: "height" } }], inputs: { media_id: "", blank_width: 0, blank_height: 0 } },
    { id: "aspect", node: "FunUIAspectSize", roles: [{ at: "project.video", input: "width" }, { at: "project.video", input: "height" }], inputs: { width: 768, height: 512 } },
    { id: "source_image", node: "FunUILoadMedia", roles: [{ at: "assets.reference_1", input: "media_id", text_fallback: { set: [{ slot: "anchor", input: "bypass", value: true }], size: { slot: "aspect", width: "width", height: "height" } } }], inputs: { media_id: "" } },
    { id: "anchor", node: "LTXVImgToVideoInplace", inputs: { bypass: false } },
    { id: "positive", node: "CLIPTextEncode", roles: [{ at: "generation.prompt", input: "text" }], inputs: { text: "" } },
  ],
  direct: [
    { id: "source_image", node: "FunUILoadMedia", roles: [{ at: "assets.source_image", input: "media_id", video_frames: true, video_only: true, text_fallback: { set: [{ slot: "sigmas", input: "sigmas", value: "1.0, 0.0" }], round_size: 32, round_frames: 8 } },
      { at: "project.video", input: "blank_width" }, { at: "project.video", input: "blank_height" }, { at: "project.video", input: "blank_frames", drives: "frames" }], inputs: { media_id: "", blank_width: 770, blank_height: 500 } },
    { id: "sigmas", node: "ManualSigmas", inputs: { sigmas: "0.9, 0.0" } },
    { id: "positive", node: "CLIPTextEncode", roles: [{ at: "generation.prompt", input: "text" }], inputs: { text: "" } },
  ],
};
const run = (slots, scene, project = {}) => buildInputs({ project: { id: "P", video: { width: 770, height: 500 }, num_frames_per_scene: 100, generation_mode: "i2v", ...project }, scene: { id: "s", text: "hi", ...scene }, slots, expand: async (x) => ({ text: x.text }), nodeInfo: async () => null, frames: 100 });

test("image to video from references: a picture in References is the start and the timeline picture gives the shape; with neither the anchor is bypassed and a plain picture stands in", async () => {
  kinds.set("pic", "image");
  let r = await run(LTX.refs, { references: ["pic"] });
  assert.equal(r.inputs.source_image.media_id, "pic");
  assert.equal(r.inputs.anchor, undefined);
  r = await run(LTX.refs, { source_image: "pic" });                       // the timeline picture is the ASPECT source here, not the start
  assert.equal(r.inputs.aspect_source.media_id, "pic");
  assert.ok(!(r.inputs.source_image || {}).media_id);
  assert.deepEqual(r.left, []);
  r = await run(LTX.refs, {});
  assert.equal(r.inputs.anchor.bypass, true);
  assert.equal(r.inputs.source_image.blank_width, 770);          // the bypassed anchor never looks at this picture
  assert.equal(r.inputs.aspect_source.blank_width, 770);        // with no aspect picture a plain one of the project's size stands in
  assert.ok(r.notes.some((n) => /text to video/.test(n)));
});

test("video to video, direct: with no video the whole schedule runs over a grey clip of the project's size, rounded for the model", async () => {
  kinds.set("vid", "video");
  let r = await run(LTX.direct, { references: ["vid"] });
  assert.equal(r.inputs.source_image.media_id, "vid");
  assert.equal(r.inputs.sigmas, undefined);
  r = await run(LTX.direct, {});
  assert.equal(r.inputs.sigmas.sigmas, "1.0, 0.0");
  assert.deepEqual([r.inputs.source_image.blank_width, r.inputs.source_image.blank_height, r.inputs.source_image.blank_frames], [768, 480, 97]);
  assert.ok(!r.notes.some((n) => /works from a video/.test(n)));
});
