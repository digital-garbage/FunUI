import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, sheetOf } from "./helper.js";

const make = () => {
  const project = { frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }, { id: "b", text: "it stops" }] };
  return { project, edit: (fn) => fn(project) };
};

test("the sheet numbers the scenes from 1 with length and prompt", () => {
  const s = sheetOf(make().project);
  assert.match(s, /Scene 1 \(4\.04 s, no references\): a dog runs/);
  assert.match(s, /Scene 2/);
});

test("allowed changes edit the prompt and length, add a scene after another, and set the default length", () => {
  const p = make();
  applyAction(p, { tool: "edit_scene", args: { scene: 2, text: "it sits", seconds: 8 } });
  assert.equal(p.project.scenes[1].text, "it sits");
  assert.equal(p.project.scenes[1].frames_mode, "custom");
  assert.ok(Math.abs(p.project.scenes[1].frames / 24 - 8) < 0.5);
  applyAction(p, { tool: "add_scene", args: { text: "a cat", after: 1 } });
  assert.deepEqual(p.project.scenes.map((s) => s.text), ["a dog runs", "a cat", "it sits"]);
  applyAction(p, { tool: "set_clip_length", args: { seconds: 6 } });
  assert.ok(Math.abs(p.project.num_frames_per_scene / 24 - 6) < 0.5);
  p.project.scenes[0].references = ["m1abcdef"]; p.project.scenes[0].source_image = "m2abcdef";
  assert.match(sheetOf(p.project), /Scene 1 .*references: m1abcdef.*aspect source: m2abcdef/);
  applyAction(p, { tool: "copy_reference", args: { scene: 3, from_scene: 1 } });
  assert.deepEqual(p.project.scenes[2].references, ["m1abcdef"]);
  assert.equal(p.project.scenes[2].source_image, "m2abcdef");
  assert.equal(applyAction(p, { tool: "edit_scene", args: { scene: 9, text: "x" } }), "no such scene");
});
