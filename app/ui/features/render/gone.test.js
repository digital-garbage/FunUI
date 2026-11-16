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

test("a render whose file is gone is forgotten with its takes; what is still on disk stays", () => {
  const p = project();
  assert.equal(forgetGone(p, new Set([keyOf(temp("a2.mp4")), keyOf(temp("a1.mp4"))])), true);
  assert.deepEqual(Object.keys(p.scene_renders), ["b"]);
  assert.deepEqual(p.scene_variants, { b: [{ media: temp("b1.mp4"), promptId: "r3" }] });
  assert.deepEqual([p.scenes[0].rating, p.scenes[0].rated, p.scenes[1].rating], ["", {}, "7"]);
  assert.equal(forgetGone(p, new Set(["temp//nothing.mp4"])), false);
});

test("ratings are kept without the files: each moves to the project's past ratings with the prompt it was given to, once", () => {
  const p = { id: "p", scenes: [{ id: "a", text: "now", rating: "8", rated: { text: "a fox", at: 5 } }, { id: "b", text: "typed", rating: "3" }, { id: "c", text: "half", cut_offset_frames: 10, rating: "8" }],
    scene_renders: { a: { media: temp("a2.mp4"), promptId: "r2", text: "a fox" }, b: { media: temp("b1.mp4"), text: "as sent" }, c: { media: temp("a2.mp4"), promptId: "r2" } },
    scene_variants: { a: [{ media: temp("a1.mp4"), promptId: "r1", rating: "9", rated: { text: "rain" } }, { media: temp("a0.mp4"), promptId: "r0", rating: "" },
      { media: temp("a2.mp4"), promptId: "r2", rating: "8", rated: { text: "a fox" } }] } };       // the last take is the one on the clip
  forgetGone(p, new Set(["a2.mp4", "a1.mp4", "a0.mp4", "b1.mp4"].map((f) => keyOf(temp(f)))));
  assert.deepEqual(p.past_ratings, [
    { rating: "8", rated: { text: "a fox", at: 5 }, promptId: "r2" },
    { rating: "3", rated: { text: "as sent" } },                              // no text was kept with the rating: the prompt the render was made from
    { rating: "9", rated: { text: "rain" }, promptId: "r1" }]);
  assert.deepEqual([p.scenes.map((s) => s.rating), p.scene_renders, p.scene_variants], [["", "", ""], {}, {}]);
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

import feature from "./gone.js";

test("a file deleted while its project is open is forgotten: after a delete in the files window, when ComfyUI comes back, and before a run", async () => {
  const said = [];
  {
    const onDisk = new Set(["a2.mp4", "a1.mp4", "b1.mp4"]);
    const heard = new Set(), subs = new Set();
    const app = { project: holder(project()), beforeRun: [], on: (fn) => { heard.add(fn); return () => heard.delete(fn); },
      api: { rendersGone: async (media) => ({ known: true, gone: media.map((m, i) => (onDisk.has(m.filename) ? -1 : i)).filter((i) => i >= 0) }) },
      generate: { subscribe: (fn) => { subs.add(fn); return () => subs.delete(fn); } } };
    const settle = () => new Promise((r) => setTimeout(r, 0));
    const off = feature.setup({ app, warn: (text) => said.push(text) });
    await settle();
    assert.deepEqual(Object.keys(app.project.project.scene_renders), ["a", "b"]);          // everything is there: nothing forgotten, nothing said
    assert.equal(said.length, 0);

    onDisk.delete("a1.mp4");                                                                // deleted in the Generated & Output Files window
    heard.forEach((fn) => fn("files.deleted"));
    await settle();
    assert.deepEqual(app.project.project.scene_variants.a.map((t) => t.media.filename), ["a2.mp4"]);
    assert.equal(said.length, 1);

    onDisk.delete("a2.mp4");                                                                // ComfyUI restarted under an open page
    subs.forEach((fn) => fn({ connection: "live" }));                                       // a first connection is not a comeback
    await settle();
    assert.deepEqual(Object.keys(app.project.project.scene_renders), ["a", "b"]);
    subs.forEach((fn) => fn({ connection: "lost" }));
    subs.forEach((fn) => fn({ connection: "live" }));
    await settle();
    assert.deepEqual(Object.keys(app.project.project.scene_renders), ["b"]);

    onDisk.delete("b1.mp4");                                                                // deleted by hand: the next run asks first
    assert.equal(app.beforeRun.length, 1);
    await app.beforeRun[0]();
    assert.deepEqual(app.project.project.scene_renders, {});
    assert.equal(said.length, 3);

    off();
    assert.deepEqual([heard.size, subs.size, app.beforeRun.length], [0, 0, 0]);
  }
});
