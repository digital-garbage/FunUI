import test from "node:test";
import assert from "node:assert/strict";
import { notTaught } from "./index.js";

const ps = (key, learning = true) => { const taste = { id: "taste" }; return { modulesById: () => ({ taste }), useful: () => learning, currentValues: () => ({ taste: { key } }) }; };

test("the not-taught reason blames a missing key only when the run captured nothing, a learner is on, and no key is set", () => {
  const nothing = { why: "this run captured nothing", reason: "nothing" }, forgotten = { why: "a new Generate started", reason: "forgotten" };
  assert.match(notTaught(nothing, ps("")), /captured nothing, and no Taste key is set/);
  assert.match(notTaught(forgotten, ps("")), /new Generate started/, "the key cleared after the run is not why");
  assert.match(notTaught(nothing, ps("", false)), /captured nothing/, "nothing learns anyway: a key would not have helped");
  assert.match(notTaught(nothing, ps("portraits")), /captured nothing/);
  assert.equal(notTaught({ recorded: ["vel"], why: null }, ps("")), "");
});

test("the scene picker lists every scene and picking one selects it like the timeline does", async () => {
  const { setupDom, teardownDom } = await import("../../composer/tests/_dom.js");
  setupDom();
  try {
    const { default: rating, sceneItems } = await import("./index.js");
    const project = { scenes: [{ id: "a", text: "first" }, { id: "b", text: "" }], scene_renders: { a: { media: { filename: "a.mp4" } } } };
    assert.deepEqual(sceneItems(project).map((i) => i.label), ["Scene 1", "Scene 2"]);
    assert.deepEqual(sceneItems(null), []);
    const picked = [];
    const app = { project: { project, selected: null, pref: () => "" , setPref() {} }, selection: { focus: "a", pick: (id) => { picked.push(id); } }, api: {}, pipeline: {}, on: () => () => {} };
    const host = document.createElement("div");
    document.body.append(host);
    rating.setup({ host, app });
    const button = host.querySelector("button");
    assert.match(button.textContent, /Scene 1/);
    button.click();
    const entries = [...document.querySelectorAll(".cx-menu-item")];
    assert.equal(entries.length, 2);
    entries[1].click();
    assert.deepEqual(picked, ["b"]);
  } finally { teardownDom(); }
});
