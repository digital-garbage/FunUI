import test from "node:test";
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
