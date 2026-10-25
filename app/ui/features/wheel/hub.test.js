import test from "node:test";
import assert from "node:assert/strict";
import { createDom } from "../../shell/dom.js";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub, readPrefs, createProject;
test.before(async () => {
  setupDom();
  globalThis.localStorage = window.localStorage;
  ({ createProject } = await import("../../shell/project.js"));
  ({ default: hub, readPrefs } = await import("./index.js"));
});
test.after(() => teardownDom());

/** A server holding two projects, as the backend does: whatever was PUT comes back on the next GET. */
function server() {
  const store = {
    aaaaaaaaaaaa: { id: "aaaaaaaaaaaa", name: "A", scenes: [], updated_at: 1 },
    bbbbbbbbbbbb: { id: "bbbbbbbbbbbb", name: "B", scenes: [], updated_at: 1 },
  };
  globalThis.fetch = async (path, opts = {}) => {
    if (path === "/fun/api/projects" && (!opts.method || opts.method === "GET")) return { ok: true, status: 200, json: async () => ({ projects: Object.values(store).map((p) => ({ id: p.id })) }) };
    const id = decodeURIComponent(String(path).split("/").pop());
    if (opts.method === "PUT") { const body = JSON.parse(opts.body); store[id] = { ...body, updated_at: 2 }; return { ok: true, status: 200, json: async () => store[id] }; }
    return { ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(store[id])) };
  };
}

test("the wheel's settings are saved with the project and come back on a fresh browser, per project", async () => {
  server();
  localStorage.clear();
  const project = createProject({});
  await project.start();                                   // project A
  const app = { dom: createDom(), project, pipeline: { allOff: () => false }, actions: [], editorSettings: [] };
  const off = hub.setup({ host: document.createElement("div"), app });
  try {
    // change it through the real settings panel
    const handle = app.editorSettings[0]();
    document.body.appendChild(handle.node); const panel = handle.node;
    const half = [...panel.querySelectorAll("button,[role=radio]")].find((b) => b.textContent.trim() === "Half wheel");
    const left = [...panel.querySelectorAll("button,[role=radio]")].find((b) => b.textContent.trim() === "Left");
    half.click(); left.click();
    await project.flush();
    assert.deepEqual(readPrefs(app), { shape: "half", side: "left", enabled: null });

    // a rental with an empty browser: same project, no localStorage
    localStorage.clear();
    const fresh = createProject({});
    await fresh.start();
    assert.deepEqual(readPrefs({ project: fresh }), { shape: "half", side: "left", enabled: null }, "the choice did not travel with the project");

    // another project keeps its own choice
    await fresh.open("bbbbbbbbbbbb");
    assert.deepEqual(readPrefs({ project: fresh }), { shape: "full", side: "right", enabled: null });
  } finally { off(); }
});

test("the wheel offers Clear references and the guidance switch, both opt-in, and they act", async () => {
  server();
  localStorage.clear();
  const project = createProject({});
  await project.start();
  let guidance = false;
  let camera = false;
  const pipeline = { allOff: () => false, cameraOff: () => camera, setCameraOff: async (on) => { camera = on; }, modulesById: () => ({ conditioning_shot_camera: {} }), guidanceOff: () => guidance, setGuidanceOff: async (on) => { guidance = on; } };
  const app = { dom: createDom(), project, pipeline, actions: [], editorSettings: [], openSettings: () => {} };
  const off = hub.setup({ host: document.createElement("div"), app });
  try {
    const find = (id) => app.actions.find((a) => a.id === id);
    assert.ok(find("guidance-off") && find("clear-refs"));
    assert.equal(find("guidance-off").optIn, true);
    assert.equal(find("guidance-off").label, "Disable all guidance enhancements");
    await find("guidance-off").run();
    assert.equal(guidance, true);
    assert.equal(find("guidance-off").label, "Enable guidance enhancements");
    assert.equal(find("camera-off").optIn, true);
    assert.equal(find("camera-off").label, "Disable camera move enhancements");
    await find("camera-off").run();
    assert.equal(camera, true);
    assert.equal(find("camera-off").label, "Enable camera move enhancements");
    assert.ok(!find("clear-refs").when(), "nothing to clear without references");
  } finally { off(); }
});

test("Quick A/B is an opt-in wheel action, offered only once there is a seed to reuse, and starts the run", async () => {
  server();
  localStorage.clear();
  const project = createProject({});
  await project.start();
  let ran = 0;
  const app = { dom: createDom(), project, pipeline: { allOff: () => false, setAllOff: async () => {} }, actions: [], editorSettings: [], lastRun: { n: 0 }, runner: { quickAB: async () => { ran += 1; } } };
  const off = hub.setup({ host: document.createElement("div"), app });
  try {
    const a = app.actions.find((x) => x.id === "quick-ab");
    assert.ok(a && a.optIn && a.label === "Quick A/B");
    assert.ok(!a.when(), "hidden until a run has made a seed");
    app.lastRun.seed = 7;
    assert.ok(a.when());
    await a.run();
    assert.equal(ran, 1);
  } finally { off(); }
});
