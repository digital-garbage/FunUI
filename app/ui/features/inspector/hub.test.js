import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { createDom } from "../../shell/dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("a scene's text changed elsewhere (the Story box) shows in the Prompt box, but not under the caret", async () => {
  let fire;
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, scenes: [{ id: "a", text: "old" }], scene_renders: {} };
  const p = { project: doc, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: () => {}, edit: (fn) => fn(doc) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { dom: createDom(), project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {} } });
  const box = () => [...host.querySelectorAll("textarea")][0];
  assert.equal(box().value, "old");
  doc.scenes[0].text = "from story";
  fire("change");
  assert.equal(box().value, "from story");
  box().focus();
  doc.scenes[0].text = "while typing";
  fire("change");
  assert.equal(box().value, "from story");          // not rebuilt under the caret
  box().blur();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(box().value, "while typing");        // caught up once the focus left
});

test("typing in the Prompt box, then pressing a button in the panel, does not rebuild the button out from under the click", async () => {
  let fire;
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, scenes: [{ id: "a", text: "old" }], scene_renders: {} };
  const p = { project: doc, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: () => {}, edit: (fn) => fn(doc) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { dom: createDom(), project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {} } });
  const box = host.querySelector("textarea");
  box.focus();
  doc.scenes[0].text = "typed here"; fire("change");                 // the box's own edit, as setText would announce it
  const button = box.closest("[aria-label=Properties]") ? [...box.closest("[aria-label=Properties]").querySelectorAll("button, input[type=checkbox]")].find((b) => b !== box) : null;
  assert.ok(button, "a control inside the panel body");
  button.focus();                                                     // the mouse-down moves focus to it
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(button.isConnected, "the pressed button is still there for its mouse-up");
  host.remove();
});

test("the scene's Video input card switches on, shows its controls, and says when the project's settings win", async () => {
  let fire;
  const calls = [];
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, scenes: [{ id: "a", text: "x" }], scene_renders: {} };
  const p = { project: doc, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: (id, key, value) => { calls.push([key, value]); doc.scenes[0][key] = value; fire("change"); }, edit: (fn) => fn(doc) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { dom: createDom(), project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {} } });
  assert.ok(host.textContent.includes("Video input"));
  assert.ok(!host.textContent.includes("Every Nth frame"));
  const sw = [...host.querySelectorAll("input[role=switch]")].find((i) => i.closest(".cx-toggle-row").textContent.includes("Prepare video inputs"));
  sw.checked = true; sw.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.equal(calls[0][0], "video_input");
  assert.equal(calls[0][1].on, true);
  assert.ok(host.textContent.includes("Every Nth frame") && host.textContent.includes("Skip first") && host.textContent.includes("Cut last"));
  doc.video_input = { on: true }; fire("change");
  assert.ok(host.textContent.includes("project's video input settings are used"));
  host.remove();
});

test("the Project tab flips width and height in one undo step", async () => {
  let fire;
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, video: { width: 768, height: 512 }, scenes: [{ id: "a", text: "x" }], scene_renders: {} };
  let edits = 0;
  const p = { project: doc, video: doc.video, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: () => {}, setField: () => {}, setVideo: () => {}, edit: (fn) => { edits++; const r = fn(doc); fire("change"); return r; } };
  const slots = [{ roles: [{ at: "project.video", input: "width", label: "Width" }, { at: "project.video", input: "height", label: "Height" }], inputs: { width: 768, height: 512 } }];
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { dom: createDom(), project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {}, pipeline: { slots: () => slots } } });
  [...host.querySelectorAll("[role=radio]")].find((b) => b.textContent === "Project").click();
  const button = () => [...host.querySelectorAll("button")].find((b) => b.textContent.includes("Flip width and height"));
  assert.ok(button(), "the flip button is on the Project tab");
  button().click();
  assert.deepEqual([doc.video.width, doc.video.height], [512, 768]);
  assert.equal(edits, 1);
  host.remove();
});
