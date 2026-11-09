import test, { mock } from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { createDom } from "../../shell/dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

// A project with one rendered scene and the Upscale button on: pressing it runs one job with the model file or with RTX.
function rig(model, upscaleJob) {
  const media = { filename: "a.mp4", subfolder: "", type: "output" };
  const doc = { id: "P", scenes: [{ id: "a", text: "x" }], scene_renders: { a: { media } }, editor_settings: { upscaleMode: "button", upscaleModel: model } };
  const project = { project: doc, get scenes() { return doc.scenes; }, selected: doc.scenes[0], edit: (fn) => fn(doc), editFor: async (_id, fn) => fn(doc),
    pref: (k, d) => (doc.editor_settings[k] ?? d), setPref() {}, onChange: () => () => {}, subscribe: () => () => {} };
  const has = new Set();
  const app = { dom: createDom(), project, selection: { focus: "a", ids: ["a"] }, has, api: { upscaleModels: async () => ({ models: [model], rtx: null }), upscale: async () => ({ job_id: "j" }), upscaleJob: upscaleJob, stopJob() {} },
    editorSettings: [], on: () => () => {}, openSettings() {} };
  const host = document.createElement("div");
  hub.setup({ host, app });
  return { app, host, has };
}

test("the upscale-model flag is up while a model job runs and down after it, also when it fails", async () => {
  let release, seen = [];
  const { host, has } = rig("4x.safetensors", () => new Promise((r) => { release = r; }));
  const press = () => host.querySelector("button");
  press().click();
  await new Promise((r) => setTimeout(r, 1300));       // the job is polled after a second
  seen.push(has.has("upscale.model"));
  release({ state: "error", detail: "boom" });
  await new Promise((r) => setTimeout(r, 100));
  seen.push(has.has("upscale.model"));
  assert.deepEqual(seen, [true, false]);
});

// "Always": a render that appears while ComfyUI is generating is tried again every 15 s, for the same project and render only.
function always(refused) {
  const first = { filename: "a.mp4", subfolder: "", type: "output" };
  const doc = { id: "P", scenes: [{ id: "a" }, { id: "b" }], scene_renders: { a: { media: first } }, editor_settings: { upscaleMode: "always", upscaleModel: "4x.safetensors" } };
  const project = { project: doc, get scenes() { return doc.scenes; }, selected: null, edit: (fn) => fn(doc), editFor: async (_id, fn) => fn(doc),
    pref: (k, d) => (doc.editor_settings[k] ?? d), setPref() {}, };
  let calls = 0, listener = () => {};
  const api = { upscaleModels: async () => ({ models: ["4x.safetensors"], rtx: null }), upscale: async () => { calls += 1; throw new Error(refused); }, upscaleJob: async () => ({}), stopJob() {} };
  const app = { dom: createDom(), project, selection: { focus: null, ids: [] }, has: new Set(), api, editorSettings: [], on: (fn) => { listener = fn; return () => {}; }, openSettings() {} };
  hub.setup({ host: document.createElement("div"), app });
  const render = (id, name) => { doc.scene_renders = { ...doc.scene_renders, [id]: { media: { filename: name, subfolder: "", type: "output" } } }; listener(); };
  return { doc, render, calls: () => calls };
}
const flush = async () => { for (let i = 0; i < 6; i += 1) await Promise.resolve(); };
const BUSY = "ComfyUI is generating right now.";

test("Always tries a render again after a running generation, for as long as it is the same render", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { render, calls } = always(BUSY);
    render("b", "b.mp4"); await flush();
    assert.equal(calls(), 1);
    mock.timers.tick(15000); await flush();
    assert.equal(calls(), 2, "tried again");
    for (let i = 0; i < 600; i += 1) { mock.timers.tick(15000); await flush(); }
    assert.equal(calls(), 481, "one first try and 480 more, then it gives up");
  } finally { mock.timers.reset(); }
});

test("Always does not try again when the render was replaced, the mode was switched off or another project is open", async () => {
  for (const change of [(d) => { d.scene_renders = { a: d.scene_renders.a }; }, (d) => { d.editor_settings.upscaleMode = "never"; }, (d) => { d.id = "Q"; }]) {
    mock.timers.enable({ apis: ["setTimeout"] });
    try {
      const { doc, render, calls } = always(BUSY);
      render("b", "b.mp4"); await flush();
      assert.equal(calls(), 1);
      change(doc);
      mock.timers.tick(15000); await flush();
      assert.equal(calls(), 1);
    } finally { mock.timers.reset(); }
  }
});
