import test from "node:test";
import assert from "node:assert/strict";
import { renderFiles, forgetGone, keyOf } from "../../shell/renders.js";
import { forgetGoneRenders } from "./gone.js";

const temp = (filename) => ({ filename, subfolder: "", type: "temp" });
const project = () => ({ id: "p", scenes: [{ id: "a", rating: "8", rated: { x: 1 } }, { id: "b", rating: "7" }],
  scene_renders: { a: { media: temp("a2.mp4"), promptId: "r2" }, b: { media: temp("b1.mp4"), promptId: "r3" } },
  scene_variants: { a: [{ media: temp("a1.mp4"), promptId: "r1", rating: "9" }, { media: temp("a2.mp4"), promptId: "r2" }], b: [{ media: temp("b1.mp4"), promptId: "r3" }] } });

test("every file a project names is listed once: the render on each clip and every take", () => {
  assert.deepEqual(renderFiles(project()).map((m) => m.filename), ["a2.mp4", "b1.mp4", "a1.mp4"]);
  assert.deepEqual(renderFiles({}), []);
});

test("a render whose file is gone is forgotten with its takes and its rating; what is still on disk stays", () => {
  const p = project();
  assert.equal(forgetGone(p, new Set([keyOf(temp("a2.mp4")), keyOf(temp("a1.mp4"))])), true);
  assert.deepEqual(Object.keys(p.scene_renders), ["b"]);
  assert.deepEqual(p.scene_variants, { b: [{ media: temp("b1.mp4"), promptId: "r3" }] });
  assert.deepEqual([p.scenes[0].rating, p.scenes[0].rated, p.scenes[1].rating], ["", {}, "7"]);
  assert.equal(forgetGone(p, new Set(["temp//nothing.mp4"])), false);
});

const holder = (open) => ({ project: open, edit(fn) { return fn(this.project); } });

test("after a restart emptied the temp folder, opening a project forgets what is gone; with no answer nothing is touched", async () => {
  const p = holder(project());
  const n = await forgetGoneRenders(p, { rendersGone: async (media) => ({ known: true, gone: media.map((m, i) => (m.filename.startsWith("a") ? i : -1)).filter((i) => i >= 0) }) });
  assert.equal(n, 2);
  assert.deepEqual(Object.keys(p.project.scene_renders), ["b"]);
  for (const api of [{ rendersGone: async () => ({ known: false, gone: [0, 1, 2] }) }, { rendersGone: async () => { throw new Error("down"); } }, { rendersGone: async () => ({ known: true, gone: [] }) }]) {
    const q = holder(project());
    assert.equal(await forgetGoneRenders(q, api), 0);
    assert.deepEqual(q.project, project());
  }
});

test("an answer that arrives after another project was opened changes nothing", async () => {
  const p = holder(project());
  const n = await forgetGoneRenders(p, { rendersGone: async () => { p.project = { id: "other", scenes: [], scene_renders: { z: { media: temp("a2.mp4") } } }; return { known: true, gone: [0] }; } });
  assert.equal(n, 0);
  assert.deepEqual(Object.keys(p.project.scene_renders), ["z"]);
});
