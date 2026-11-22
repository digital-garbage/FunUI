import test from "node:test";
import assert from "node:assert/strict";
import { applyAction, sheetOf } from "./window.js";

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

import { catalogOf } from "./window.js";

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
  const p = { project, setAnchor: (v) => { project.anchor = v; }, setPostfix: (v) => { project.postfix = v; }, setPostfixEnabled: (v) => { project.postfix_enabled = v; }, setNegative: (v) => { project.negative = v; }, setField: (k, v) => { project[k] = v; } };
  const api = { saveShortcut: async (...a) => { calls.push(["save", ...a]); }, deleteShortcut: async (n) => { calls.push(["delete", n]); } };
  assert.match(sheetOf(project), /scene prompt\): film still[\s\S]*\$dog = a collie/);
  applyAction(p, { tool: "set_project_text", args: { field: "anchor", text: "cinema" } });
  applyAction(p, { tool: "set_variable", args: { name: "cat", value: "a tabby" } });
  applyAction(p, { tool: "set_variable", args: { name: "dog", value: "a pug" } });
  applyAction(p, { tool: "remove_variable", args: { name: "cat" } });
  assert.equal(project.anchor, "cinema");
  assert.deepEqual(project.variables, [{ name: "dog", value: "a pug" }]);
  await applyAction(p, { tool: "save_shortcut", args: { name: "sunset", triggers: ["sunset"], replacements: ["golden light"], existing: true } }, null, api);
  await applyAction(p, { tool: "remove_shortcut", args: { name: "old" } }, null, api);
  assert.deepEqual(calls, [["save", { name: "sunset", triggers: ["sunset"], replacements: ["golden light"], enabled: true, category: "", sub_category: "" }, "sunset"], ["delete", "old"]]);
});

import { inputsOf } from "./window.js";

test("pipeline inputs are listed with their choices, set together in one save, and a preset replaces the pipeline", async () => {
  const saved = [], restored = [], prefs = {};
  const ps = { slots: () => [{ id: "s1", node: "FunUISampler", group: "Sampling", inputs: { steps: 20, model: ["x", 0] } }],
    save: async (b) => { saved.push(b); }, restore: async (b) => { restored.push(b); return { refused: [] }; } };
  const app = { pipeline: ps, api: { describeNodes: async () => ({ nodes: { FunUISampler: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100, default: 20 }, { name: "sampler_name", type: "COMBO", choices: ["euler", "seeds_2"], default: "euler" }, { name: "model", type: "COMBO", choices: ["a"] }] } } }) } };
  const list = await inputsOf(app);
  assert.deepEqual(list.map((x) => [x.slot, x.name, x.now]), [["s1", "steps", 20], ["s1", "sampler_name", "euler"]]);       // a wired input is not a value
  const p = { project: { scenes: [] }, setPref: (k, v) => { prefs[k] = v; }, setField: (k, v) => { prefs[k] = v; } };
  await applyAction(p, { tool: "set_node_inputs", args: { changes: [{ slot: "s1", input: "steps", value: 5 }, { slot: "s1", input: "sampler_name", value: "seeds_2" }] } }, ps);
  assert.deepEqual(saved, [{ inputs: { s1: { steps: 5, sampler_name: "seeds_2" } } }]);
  await applyAction(p, { tool: "apply_preset", args: { preset: "fast" }, preset: { title: "H3 fast", slots: [{ id: "n" }], prefs: { a: 1 }, project: { b: 2 } } }, ps);
  assert.deepEqual(restored, [{ slots: [{ id: "n" }], removed: [], unwired: {} }]);
  assert.deepEqual(prefs, { a: 1, b: 2 });
});

test("cuts: the sheet says which scenes are one clip, cut_scene splits a scene, set_cut_words saves the words", async () => {
  const saved = [];
  const project = { frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }, { id: "b", text: "" }] };
  const p = { project, edit: (fn) => fn(project) };
  assert.match(sheetOf(project), /Scene 1 /);
  applyAction(p, { tool: "cut_scene", args: { scene: 1 } });
  assert.equal(project.scenes.length, 3);
  assert.match(sheetOf(project), /Scene 1 .*one clip with scenes 1 and 2/);
  await applyAction(p, { tool: "set_cut_words", args: { words: ["qcut", "scene"] } }, null, { saveStoryMarkers: async (w) => { saved.push(w); } });
  assert.deepEqual(saved, [["qcut", "scene"]]);
});

test("the postfix switch is changed through the same tool", () => {
  const project = { scenes: [], postfix_enabled: true };
  applyAction({ project, setPostfixEnabled: (v) => { project.postfix_enabled = v; } }, { tool: "set_project_text", args: { field: "postfix_enabled", text: false } });
  assert.equal(project.postfix_enabled, false);
});

test("set_story splits the new text at the cut word and rewrites the scenes in order", async () => {
  const project = { frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "one", source: { type: "carry" } }, { id: "b", text: "two", source: { type: "carry" } }] };
  const p = { project, edit: (fn) => fn(project) };
  const api = { storySplit: async (text) => ({ scenes: text.split("\nqcut\n") }) };
  const out = await applyAction(p, { tool: "set_story", args: { text: "uno\nqcut\ndos\nqcut\ntres" } }, null, api);
  assert.match(out, /3 scenes/);
  assert.deepEqual(project.scenes.map((s) => s.text), ["uno", "dos", "tres"]);
});

test("run_action picks the named scenes, starts the app's action without waiting, and refuses one that is not offered", async () => {
  const picked = [], ran = [];
  const project = { scenes: [{ id: "a" }, { id: "b" }, { id: "c" }] };
  const app = { selection: { pick: (id, o) => picked.push([id, Boolean(o.additive)]) }, actions: [{ id: "generate-selected", label: "Generate selected", run: () => { ran.push("go"); } }, { id: "hidden", label: "Hidden", when: () => false, run: () => ran.push("no") }] };
  const out = applyAction({ project }, { tool: "run_action", args: { action: "generate-selected", scenes: [2, 3] } }, null, null, app);
  assert.equal(out, "Started: Generate selected");
  await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual(picked, [["b", false], ["c", true]]);
  assert.deepEqual(ran, ["go"]);
  assert.equal(applyAction({ project }, { tool: "run_action", args: { action: "hidden" } }, null, null, app), "that action is not available now");
});

test("a scene can get its own prefix and go back to the project's", () => {
  const p = make();
  applyAction(p, { tool: "edit_scene", args: { scene: 2, prefix: " warm light " } });
  assert.equal(p.project.scenes[1].anchor, "warm light");
  assert.match(sheetOf(p.project), /Scene 2 .*own prefix: warm light/);
  applyAction(p, { tool: "edit_scene", args: { scene: 2, prefix: "" } });
  assert.equal(p.project.scenes[1].anchor, "");                       // no prefix at all for this scene
  applyAction(p, { tool: "edit_scene", args: { scene: 2, project_prefix: true } });
  assert.equal("anchor" in p.project.scenes[1], false);
});

test("pinned references show in the sheet, share_reference gives every scene the same ones, copy_reference picks the scene so the media bin shows it", () => {
  const picked = [];
  const project = { frame_rate: 24, num_frames_per_scene: 97, fixed_references: { pin1: 1 }, scenes: [{ id: "a", text: "x", references: ["r1", "r2"] }, { id: "b", text: "y" }, { id: "c", text: "z", references: ["r1"] }] };
  const p = { project, edit: (fn) => fn(project) };
  assert.match(sheetOf(project), /Scene 2 .*references: pin1 \(pinned to every scene\)/);
  const out = applyAction(p, { tool: "share_reference", args: { from_scene: 1 } });
  assert.match(out, /2 scenes got the references of scene 1/);
  assert.deepEqual(project.scenes[1].references, ["r1", "r2"]);
  assert.deepEqual(project.scenes[2].references, ["r1", "r2"]);
  applyAction(p, { tool: "copy_reference", args: { scene: 2, from_scene: 3 } }, null, null, { selection: { pick: (id) => picked.push(id) } });
  assert.deepEqual(picked, ["b"]);
});

test("a scene the helper adds gets the first scene's references and aspect source", () => {
  const p = make();
  p.project.scenes[0].references = ["m1abcdef", "m3abcdef"]; p.project.scenes[0].source_image = "m2abcdef";
  applyAction(p, { tool: "add_scene", args: { text: "a cat" } });
  const added = p.project.scenes[p.project.scenes.length - 1];
  assert.deepEqual(added.references, ["m1abcdef", "m3abcdef"]);
  assert.equal(added.source_image, "m2abcdef");
  added.references.push("x");
  assert.deepEqual(p.project.scenes[0].references, ["m1abcdef", "m3abcdef"]);       // a copy, not the same list
});
