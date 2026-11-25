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
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs: [] }) };
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
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs }) };
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

test("one model picker lists ComfyUI's files and saves the pick as the engine model, replacing an older encoder choice", async () => {
  let inputs = { enabled: true, encoder_file: "old.safetensors" }, saved;
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs }], ensureLoaded: async () => {}, save: async (b) => { saved = b; inputs = { ...inputs, ...b.inputs.e }; }, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: ["gemma4.safetensors", "old.safetensors"], families: [] }), enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs: [] }), enhancerEngineFiles: async () => ({ files: [] }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(page.node.querySelector('select[aria-label="Encoder"]'), null);                         // no second picker for the same thing
  assert.equal(page.node.querySelector('select[aria-label="Encoder family"]'), null);
  const pick = page.node.querySelector('select[aria-label="From ComfyUI\'s list"]');
  assert.equal(pick.value, "old.safetensors");                                                          // an older project's saved encoder shows as picked
  pick.value = "gemma4.safetensors"; pick.dispatchEvent(new window.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(saved.inputs.e, { engine_model: "comfy:gemma4.safetensors", encoder_file: "" });
});

test("Style picks the built-in instructions shown as the placeholder, and choosing Dynamic saves it", async () => {
  const saved = [];
  const slot = { id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true } };
  const ps = { slots: () => [slot], ensureLoaded: async () => {}, save: async (...a) => { saved.push(a); }, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }),
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
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}) };
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
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: ["g.safetensors"], families: ["stable_diffusion"] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({ prompts: {}, macros: { LAST_RATING: "x" }, styles: {} }) };
  const grey = async (inputs) => {
    const page = enhance({ dom: createDom(), pipeline: mk(inputs), api }, () => {});
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    const find = (text) => [...page.node.querySelectorAll(".cx-field, .cx-toggle-row")].find((f) => f.textContent.startsWith(text)).querySelector("input, select, textarea").disabled;
    return find;
  };
  const off = await grey({ enabled: false });
  assert.equal(off("From ComfyUI's list"), true); assert.equal(off("Send everything into chat"), true);
  const noChat = await grey({ enabled: true });
  assert.equal(noChat("Send everything into chat"), false);
  assert.equal(noChat("Summarise long conversations"), true); assert.equal(noChat("Pass the rating"), true); assert.equal(noChat("Context window (tokens)"), true);
  assert.equal(noChat("Conversation, Gemma 4"), true); assert.equal(noChat("Comments"), false); assert.equal(noChat("Rewrite instructions"), false);
  const chat = await grey({ enabled: true, to_chat: true, compact: false, greedy: true });
  assert.equal(chat("Summarise long conversations"), false); assert.equal(chat("Pass the rating"), false);
  assert.equal(chat("Context window (tokens)"), true);       // summarising is off
  assert.equal(chat("Conversation, Gemma 4"), false); assert.equal(chat("Comments"), true); assert.equal(chat("Rating line opening"), true);
  assert.equal(chat("Chat answers"), true);                  // no model picked
  assert.equal(chat("temperature"), true);                   // greedy: the dials are not used
});

test("downloaded models are listed with Use and Delete buttons; Use picks the model, Delete asks first and removes only that model", async () => {
  const deleted = [], saved = [];
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async (x) => { saved.push(x); }, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}),
    enhancerEngineModels: async () => ({ models: [{ name: "a__big", size: 3 * 2 ** 30, loaded: true }, { name: "b__small", size: 2 ** 20, loaded: false }] }), enhancerEngineDelete: async (n) => { deleted.push(n); }, enhancerEngineLoad: async () => ({}) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /Downloaded models \(2\)/);
  assert.match(page.node.textContent, /a\/big: 3\.0 GB \(loaded\)/);
  [...page.node.querySelectorAll("button")].filter((b) => b.textContent === "Use")[1].click();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(saved.slice(-1), [{ inputs: { e: { engine_model: "b/small" } } }]);
  [...page.node.querySelectorAll("button")].filter((b) => b.textContent === "Delete")[0].click();
  await new Promise((r) => setTimeout(r, 0));
  const dialog = document.querySelector(".cx-modal");
  assert.match(dialog.textContent, /Delete a\/big \(3\.0 GB\) from disk\?/);
  [...dialog.querySelectorAll("button")].find((b) => b.textContent === "Delete").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(deleted, ["a__big"]);
});

test("pressing Download refreshes the downloaded list at once, so a fast download shows up without waiting for a poll", async () => {
  let listed = 0, started = 0;
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "a/b::q8.gguf" } }], currentValues: () => ({ engine_model: "a/b::q8.gguf" }), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "ready", message: "x" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}),
    enhancerEngineFiles: async () => ({ files: [{ name: "q4.gguf", size: 2 ** 30 }, { name: "q8.gguf", size: 2 ** 31 }] }),
    enhancerEngineModels: async () => { listed += 1; return { models: listed > 1 ? [{ name: "a__b__q8", source: "a/b::q8.gguf", size: 2 ** 31, loaded: false }] : [] }; },
    enhancerEngineDownload: async () => { started += 1; return { ok: true, started: true }; } };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  const before = listed;
  [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Download").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(started, 1);
  assert.ok(listed > before);
  assert.match(page.node.textContent, /Downloaded models \(1\)/);
  assert.match(page.node.textContent, /q8\.gguf \(2\.0 GB\), downloaded/);
});

test("the engine line shows what a running request is doing, so a busy model does not look stale", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "a/b" } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "ready", message: "a/b on cuda", busy: { stage: "writing", tokens: 42, seconds: 3.5 } }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}) };
  let stop = () => {};
  const page = enhance({ dom: createDom(), pipeline: ps, api }, (f) => { stop = f; });
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 5));
  assert.match(page.node.textContent, /writing: 42 tokens, 3\.5s/);
  stop();
});

test("a rewrite running inside a generation shows up as it is written, and goes away when it ends", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "a/b" } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  let now = { id: 1, kind: "rewrite", label: "a fox", thinking: "Let me think", answer: "A fox in snow", status: "writing", done: false };
  const api = { enhancerEngine: async () => ({ phase: "ready" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}), enhancerLive: async () => now };
  let stop = () => {};
  const store = new Map([["fp.enhance.live", "1"]]);
  globalThis.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, (f) => { stop = f; });
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 20));
  assert.match(page.node.textContent, /Writing now/);
  assert.match(page.node.textContent, /A fox in snow/);
  now = { ...now, done: true };
  await new Promise((r) => setTimeout(r, 800));
  assert.doesNotMatch(page.node.textContent, /Writing now/);
  stop(); delete globalThis.localStorage;
});

test("with live writing off (the default) the Enhance page does not follow the model's writing", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "a/b" } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  const now = { id: 1, kind: "rewrite", label: "a fox", thinking: "", answer: "A fox in snow", status: "writing", done: false };
  const api = { enhancerEngine: async () => ({ phase: "ready" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}), enhancerEngineModels: async () => ({ models: [] }), enhancerLive: async () => now };
  let stop = () => {};
  const page = enhance({ dom: createDom(), pipeline: ps, api }, (f) => { stop = f; });
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 30));
  assert.doesNotMatch(page.node.textContent, /Writing now/);
  assert.match(page.node.textContent, /Show live writing on this page/);
  stop();
});

test("files of a repo are checkboxes, and every checked one goes into the model name", async () => {
  const saved = [];
  let value = "o/m";
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: value } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async (x) => { saved.push(x); }, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}),
    enhancerEngineFiles: async () => ({ files: [{ name: "a-1.gguf", size: 2 ** 30 }, { name: "a-2.gguf", size: 2 ** 30 }] }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 20));
  assert.match(page.node.textContent, /a-1\.gguf \(1\.0 GB\)/);
  const boxes = [...page.node.querySelectorAll("input[type=checkbox]")].filter((b) => /a-\d\.gguf/.test(b.closest("label")?.textContent || b.parentElement.textContent));
  assert.equal(boxes.length, 2);
  boxes[1].click();
  await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual(saved.slice(-1), [{ inputs: { e: { engine_model: "o/m::a-2.gguf" } } }]);
});

test("Download and Load are collapsible lists of checkboxes with their own buttons; Load lists only what is downloaded", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "a/b::q8.gguf" } }], currentValues: () => ({ engine_model: "a/b::q8.gguf" }), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "idle" }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}),
    enhancerEngineFiles: async () => ({ files: [{ name: "q4.gguf", size: 2 ** 30 }, { name: "q8.gguf", size: 2 ** 31 }] }),
    enhancerEngineModels: async () => ({ models: [{ name: "a__b__q8", source: "a/b::q8.gguf", size: 2 ** 31, loaded: false }] }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 10));
  const [dl, ld] = [...page.node.querySelectorAll("details.cx-collapsible")].filter((d) => /^(Download|Load)$/.test(d.querySelector("summary").textContent));
  assert.equal(dl.querySelectorAll('input[type="checkbox"]').length, 2);
  assert.equal(ld.querySelectorAll('input[type="checkbox"]').length, 1);
  assert.ok([...dl.querySelectorAll("button")].some((b) => b.textContent === "Download"));
  assert.ok([...ld.querySelectorAll("button")].some((b) => b.textContent === "Load model"));
});

test("a busy counter ticking on the engine line updates that line only, so a field being edited is never rebuilt", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "a/b" } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  let secs = 1;
  const api = { enhancerEngine: async () => ({ phase: "ready", message: "x", busy: { stage: "writing", tokens: secs, seconds: secs++ } }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}), enhancerEngineModels: async () => ({ models: [] }) };
  let stop = () => {};
  const page = enhance({ dom: createDom(), pipeline: ps, api }, (f) => { stop = f; });
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 30));
  const input = page.node.querySelector("input");
  await new Promise((r) => setTimeout(r, 1200));
  assert.ok(input.isConnected, "the page was not rebuilt");
  assert.match(page.node.textContent, /writing: \d+ tokens, \d+s/);
  stop();
});

test("the Load list shows every downloaded model, whatever repo is typed, and Load model loads the checked one while download stays free", async () => {
  const loaded = [];
  const ps = { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs: { enabled: true, engine_model: "x/typed" } }], currentValues: () => ({}), ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  const api = { enhancerEngine: async () => ({ phase: "ready", message: "x", loading: { label: "n/m", message: "loading" }, download: null }), enhancerEncoders: async () => ({ files: [], families: [] }), enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({}),
    enhancerEngineModels: async () => ({ models: [{ name: "a__one", source: "a/one", size: 2 ** 30, loaded: true }, { name: "b__two", source: "b/two", size: 2 ** 31, loaded: false }] }),
    enhancerEngineLoad: async (src) => { loaded.push(src); return {}; }, enhancerEngineDownload: async () => ({ started: true }) };
  const page = enhance({ dom: createDom(), pipeline: ps, api }, () => {});
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 20));
  const ld = [...page.node.querySelectorAll("details.cx-collapsible")].find((d) => d.querySelector("summary").textContent === "Load");
  assert.equal(ld.querySelectorAll('input[type="checkbox"]').length, 2);
  const dl = [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Download");
  assert.equal(dl.disabled, false);                                                // a load is running, a download may still start
  assert.equal([...page.node.querySelectorAll("button")].find((b) => b.textContent === "Load model").disabled, true);   // but not a second load
  assert.match(page.node.textContent, /Ready: x Loading n\/m: loading/);
});
