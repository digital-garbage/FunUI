import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { createDom } from "../../shell/dom.js";

let enhance;
test.before(async () => { setupDom(); ({ enhance } = await import("./enhance.js")); });
test.after(() => teardownDom());

test("the Enhance tab follows the open project's pipeline: no enhancer there, no toggle shown", async () => {
  let live = [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true } }], heard;
  const ps = { slots: () => live, ensureLoaded: async () => {}, save: async () => {}, subscribe: (fn) => { heard = fn; return () => {}; } };
  const api = { enhancerEncoders: async () => ({ files: [], families: [] }), enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs: [] }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  assert.match(page.node.textContent, /Enhance prompt/);
  live = [{ id: "s", node: "FunUISampler", inputs: {} }];              // another project opened
  heard(live);
  assert.match(page.node.textContent, /No prompt enhancer here/);
});

test("a run that ends while Enhance is open shows its final prompt without reopening the tab", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: {} }], ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  let runs = [], tell;
  const api = { enhancerEncoders: async () => ({ files: [], families: [] }), enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs }) };
  const generate = { subscribe: (fn) => { tell = fn; fn({ phase: "idle" }); return () => {}; } };
  const page = enhance({ dom: createDom(), pipeline: ps, api, generate }, () => {});
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /Nothing yet/);
  runs = [{ prompt_id: "p", status: "off: the prompt ran as typed", before: "a red fox", after: "a red fox at dusk" }];
  tell({ phase: "running" }); tell({ phase: "done" });
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /a red fox at dusk/);
});

test("Chat on a pipeline with no enhancer says so instead of taking comments that would reach nothing", async () => {
  const { chat } = await import("./chat.js");
  let slots = [{ id: "s", node: "Sampler" }], heard;
  const app = { dom: createDom(), project: { project: { id: "P", scenes: [] }, pref: (_k, d) => d, setPref: () => {} }, selection: {}, lastRun: { n: 0 },
    api: { enhancerRuns: async () => ({ runs: [] }) }, on: () => () => {}, pipeline: { slots: () => slots, subscribe: (fn) => { heard = fn; return () => {}; } } };
  const page = chat(app, () => {});
  assert.match(page.node.textContent, /No prompt enhancer here/);
  assert.equal(page.node.querySelector("textarea"), null);
  slots = [{ id: "e", node: "FunUIEnhancePrompt" }];
  heard(slots, "edit");
  assert.ok(page.node.querySelector("textarea"));
});

test("the Encoder pick lists the encoder files, and its family is greyed out until a file is picked", async () => {
  let inputs = { enabled: true }, saved;
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs }], ensureLoaded: async () => {}, save: async (b) => { saved = b; inputs = { ...inputs, ...b.inputs.e }; }, subscribe: () => () => {} };
  const api = { enhancerEncoders: async () => ({ files: ["gemma4.safetensors"], families: ["stable_diffusion", "ltxv"] }), enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs: [] }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /Encoder/);
  assert.match(page.node.textContent, /gemma4\.safetensors/);
  assert.equal(page.node.querySelector('select[aria-label="Encoder family"]').disabled, true);
  inputs = { enabled: true, encoder_file: "gemma4.safetensors" };
  const ps2 = { ...ps, slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs }] };
  const again = enhance({ dom: createDom(), pipeline: ps2, api }, () => {});
  document.body.append(again.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(again.node.querySelector('select[aria-label="Encoder family"]').disabled, false);
});

test("Style picks the built-in instructions shown as the placeholder, and choosing Dynamic saves it", async () => {
  const saved = [];
  const slot = { id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true } };
  const ps = { slots: () => [slot], ensureLoaded: async () => {}, save: async (...a) => { saved.push(a); }, subscribe: () => () => {} };
  const api = { enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }),
    enhancerDefaults: async () => ({ instructions: "DEFAULT TEXT", reference_intro: "", styles: { default: "DEFAULT TEXT", dynamic: "DYNAMIC TEXT" } }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /Style/);
  assert.match(page.node.textContent, /Dynamic \(more motion\)/);
  const area = [...page.node.querySelectorAll("textarea")].find((t) => t.placeholder === "DEFAULT TEXT");
  assert.ok(area, "the default style shows the default instructions");
});

test("the last-frame toggle is greyed out unless Continuity is loaded and carrying the frame", async () => {
  const saved = [];
  const mk = (continuity) => ({ slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true } }], currentValues: () => (continuity === undefined ? {} : { continuity }),
    ensureLoaded: async () => {}, save: async (b) => { saved.push(b); }, subscribe: () => () => {} });
  const api = { enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}) };
  const shows = async (continuity) => {
    const page = enhance({ dom: createDom(), pipeline: mk(continuity), api }, () => {});
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    return page;
  };
  const label = /Pass the last frame of previous scene as enhancer input/;
  const greyed = (page) => [...page.node.querySelectorAll('input[role="switch"]')].find((i) => label.test(i.closest(".cx-toggle-row").textContent)).disabled;
  assert.equal(greyed(await shows(undefined)), true);        // module not loaded
  assert.equal(greyed(await shows({ carry: false })), true);  // not carrying the frame
  assert.equal(greyed(await shows({ carry: true })), false);
  assert.ok(saved.length === 0);
});

test("settings that hang on another switch follow it and are greyed out while it is off", async () => {
  const mk = (inputs) => ({ slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} });
  const api = { enhancerEncoders: async () => ({ files: ["g.safetensors"], families: ["stable_diffusion"] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({ prompts: {}, macros: { LAST_RATING: "x" }, styles: {} }) };
  const grey = async (inputs) => {
    const page = enhance({ dom: createDom(), pipeline: mk(inputs), api }, () => {});
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    const find = (text) => [...page.node.querySelectorAll(".cx-field, .cx-toggle-row")].find((f) => f.textContent.startsWith(text)).querySelector("input, select, textarea").disabled;
    return find;
  };
  const off = await grey({ enabled: false });
  assert.equal(off("Encoder"), true); assert.equal(off("Send everything into chat"), true);
  const noChat = await grey({ enabled: true });
  assert.equal(noChat("Send everything into chat"), false);
  assert.equal(noChat("Summarise long conversations"), true); assert.equal(noChat("Pass the rating"), true); assert.equal(noChat("Context window (tokens)"), true);
  assert.equal(noChat("Conversation, Gemma 4"), true); assert.equal(noChat("Comments"), false); assert.equal(noChat("Rewrite instructions"), false);
  const chat = await grey({ enabled: true, to_chat: true, compact: false, greedy: true });
  assert.equal(chat("Summarise long conversations"), false); assert.equal(chat("Pass the rating"), false);
  assert.equal(chat("Context window (tokens)"), true);       // summarising is off
  assert.equal(chat("Conversation, Gemma 4"), false); assert.equal(chat("Comments"), true); assert.equal(chat("Rating line opening"), true);
  assert.equal(chat("Chat answers"), true);                  // no encoder of its own picked
  assert.equal(chat("temperature"), true);                   // greedy: the dials are not used
});
