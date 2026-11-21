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

import { catalogOf } from "./helper.js";

test("the catalog lists the settings of the active modules with their value now, and set_setting writes through the pipeline", () => {
  const set = [];
  const ps = { activeModules: () => [{ id: "sampling_attention_bandit", title: "Attention explorer", settings: { enabled: { type: "bool", default: false, label: "Attention explorer" }, blocks: { type: "string", default: "2-10", model_blocks: true } } }],
    currentValues: () => ({ sampling_attention_bandit: { enabled: true } }), modulesById: () => ({ sampling_attention_bandit: { settings: { blocks: { type: "string", model_blocks: true } } } }), blocks: () => 50,
    setModuleValue: (...a) => set.push(a) };
  const cat = catalogOf(ps);
  assert.deepEqual(cat.map((x) => [x.name, x.now]), [["enabled", true], ["blocks", "2-10"]]);
  applyAction({ project: { scenes: [] } }, { tool: "set_setting", args: { module: "sampling_attention_bandit", setting: "blocks", value: "2-10,29-90" } }, ps);
  assert.deepEqual(set, [["sampling_attention_bandit", "blocks", "2-10,29-49"]]);
});

test("prefix, postfix, variables and shortcuts: the sheet shows them and allowed changes reach the project and the library", async () => {
  const calls = [];
  const project = { anchor: "film still", postfix: "", negative: "", variables: [{ name: "dog", value: "a collie" }], scenes: [] };
  const p = { project, setAnchor: (v) => { project.anchor = v; }, setPostfix: (v) => { project.postfix = v; }, setNegative: (v) => { project.negative = v; }, setField: (k, v) => { project[k] = v; } };
  const api = { saveShortcut: async (...a) => { calls.push(["save", ...a]); }, deleteShortcut: async (n) => { calls.push(["delete", n]); } };
  assert.match(sheetOf(project), /Prefix: film still[\s\S]*\$dog = a collie/);
  applyAction(p, { tool: "set_project_text", args: { field: "prefix", text: "cinema" } });
  applyAction(p, { tool: "set_variable", args: { name: "cat", value: "a tabby" } });
  applyAction(p, { tool: "set_variable", args: { name: "dog", value: "a pug" } });
  applyAction(p, { tool: "remove_variable", args: { name: "cat" } });
  assert.equal(project.anchor, "cinema");
  assert.deepEqual(project.variables, [{ name: "dog", value: "a pug" }]);
  await applyAction(p, { tool: "save_shortcut", args: { name: "sunset", triggers: ["sunset"], replacements: ["golden light"], existing: true } }, null, api);
  await applyAction(p, { tool: "remove_shortcut", args: { name: "old" } }, null, api);
  assert.deepEqual(calls, [["save", { name: "sunset", triggers: ["sunset"], replacements: ["golden light"], enabled: true, category: "", sub_category: "" }, "sunset"], ["delete", "old"]]);
});
