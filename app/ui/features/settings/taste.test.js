import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let taste, ratedIn;
test.before(async () => { setupDom(); ({ taste, ratedIn } = await import("./taste.js")); });
test.after(() => teardownDom());
const tick = () => new Promise((r) => setTimeout(r, 0));

test("the panel shows block influence: the switch, the clip counts, and each pair against chance", async () => {
  const sets = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "fox", enabled: true, problem: null }),
    blockInfluenceGroups: async () => ({ key: "fox", used: 8, counts: { liked: 2, image: 2, composition: 2, both: 2 },
      cos: { "image vs composition": 0.5 }, chance: { "image vs composition": { mean: 0.02, p95: 0.4 } } }),
    setBlockInfluence: async (on) => { sets.push(on); return { enabled: on }; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  const text = page.node.textContent;
  assert.match(text, /Record block influence/);
  assert.match(text, /liked 2, image 2, composition 2, both 2/);
  assert.match(text, /image vs composition/);
  assert.match(text, /\+0\.50/);
  assert.match(text, /95th \+0\.40/);
  const toggle = page.node.querySelector('input[type="checkbox"]');
  assert.equal(toggle.checked, true);
  toggle.click(); await tick(); await tick();
  assert.deepEqual(sets, [false]);
});

test("nothing recorded yet is said, not shown as a comparison", async () => {
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "none", enabled: false, problem: null }),
    blockInfluenceGroups: async () => ({ key: "none", used: 0, counts: {}, cos: {}, chance: {}, note: "nothing recorded yet" }),
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  assert.match(page.node.textContent, /nothing recorded yet/);
});

test("a failed report still shows the switch, so recording can be turned on", async () => {
  const sets = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "fox", enabled: true, problem: null }),
    blockInfluenceGroups: async () => { throw new Error("HTTP 500"); },
    setBlockInfluence: async (on) => { sets.push(on); return { enabled: on }; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  const toggle = page.node.querySelector('input[type="checkbox"]');
  assert.ok(toggle, "the switch is still there");
  toggle.click(); await tick(); await tick();
  assert.deepEqual(sets, [false]);
  assert.match(page.node.textContent, /HTTP 500/);
});

test("the block influence readout shows flatness, and Clear asks before it clears", async () => {
  const cleared = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "fox", enabled: true, problem: null, runs: 9, skipped: 1, used: 8, flatness: 0.123, mean_novelty: 0.4 }),
    blockInfluenceGroups: async () => ({ key: "fox", used: 0, counts: {}, cos: {}, chance: {}, note: "nothing recorded yet" }),
    blockInfluenceExportUrl: (k) => `/x/${k}`,
    clearBlockInfluence: async (k) => { cleared.push(k); return {}; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  assert.match(page.node.textContent, /0\.123/);
  assert.match(page.node.textContent, /0\.400/);
  assert.match(page.node.textContent, /Download/);
  assert.match(page.node.textContent, /Clear/);
});

test("a failed status request still shows the switch, so recording can be turned on", async () => {
  const sets = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => { throw new Error("HTTP 500"); },
    blockInfluenceGroups: async () => ({ key: "fox", used: 0, counts: {}, cos: {}, chance: {}, note: "nothing recorded yet" }),
    setBlockInfluence: async (on) => { sets.push(on); return { enabled: on }; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  const toggle = page.node.querySelector('input[type="checkbox"]');
  assert.ok(toggle, "the switch is still there");
  toggle.click(); await tick(); await tick();
  assert.deepEqual(sets, [true], "recording is opt in: the unknown state shows off, so one click turns it on");
});

const rated = () => ({ name: "Fox", scenes: [{ id: "a", rating: "liked", rated: { at: 1 } }, { id: "b", rating: "" }],
  scene_renders: { a: { promptId: "p1" }, b: { media: {} } }, scene_variants: { a: [{ promptId: "p0", rating: "disliked", rated: { at: 2 } }] } });

test("ratedIn lists the run ids of a project's renders and takes, and counts its labels", () => {
  const { ids, labelled } = ratedIn(rated());
  assert.deepEqual(ids.sort(), ["p0", "p1"]);
  assert.equal(labelled, 2);
  assert.deepEqual(ratedIn(null), { ids: [], labelled: 0 });
});

test("Clear saved ratings asks first, forgets each clip of this project, and removes the labels", async () => {
  const sent = [], project = rated();
  const app = { project: { project, edit: (fn) => fn(project) }, api: { tasteKeys: async () => ({ keys: [] }),
    rateTaste: async (id, rating, axis) => { sent.push([id, rating, axis]); return {}; } } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  const clear = [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Clear");
  assert.ok(clear && !clear.disabled);
  clear.click(); await tick();
  assert.deepEqual(sent, [], "nothing is cleared before the person confirms");
  const confirm = document.querySelectorAll(".cx-modal-foot button")[1];
  confirm.click(); await tick(); await tick(); await tick();
  assert.deepEqual(sent.map((x) => x.join(",")).sort(), ["p0,,", "p1,,"]);
  assert.equal(project.scenes[0].rating, "");
  assert.deepEqual(project.scenes[0].rated, {});
  assert.equal(project.scene_variants.a[0].rating, "");
});

test("Clear saved ratings is off when the project holds none", async () => {
  const project = { scenes: [{ id: "a", rating: "" }], scene_renders: {}, scene_variants: {} };
  const page = taste({ project: { project }, api: { tasteKeys: async () => ({ keys: [] }) } })();
  document.body.append(page.node);
  await tick(); await tick();
  const clear = [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Clear");
  assert.ok(clear.disabled);
});
