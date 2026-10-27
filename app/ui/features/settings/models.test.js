import test from "node:test";
import assert from "node:assert/strict";
import { createDom } from "../../shell/dom.js";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let models;
test.before(async () => { setupDom(); ({ models } = await import("./models.js")); });
test.after(() => teardownDom());

test("Revert goes back to the pipeline of the project now open, not the one open when the page was", async () => {
  let live = "A", heard, restored;
  const ps = { slots: () => [{ id: "model", node: "L", inputs: { file: live } }], snapshot: () => ({ slots: [{ id: "model", node: "L", inputs: { file: live } }] }),
    ensureLoaded: async () => {}, subscribe: (fn) => { heard = fn; return () => {}; }, restore: async (snap) => { restored = snap.slots[0].inputs.file; return { refused: [] }; },
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [], queueable: () => true, removedIds: () => [], offered: () => [] };
  const page = models({ dom: createDom(), pipeline: ps, api: { describeNodes: async () => ({ nodes: { L: { widgets: [] } } }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  live = "B";
  heard(ps.slots(), "load");
  await new Promise((r) => setTimeout(r, 0));
  [...page.node.querySelectorAll("button")].find((b) => b.textContent.includes("Revert")).click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(restored, "B");
});

test("a node the server says needs something shows the orange dot, whichever way the message names it", async () => {
  const slots = [{ id: "model", node: "L", inputs: { ckpt_name: "" } }, { id: "s", node: "S", inputs: {}, roles: [{ at: "project.video", input: "amount", label: "Length" }] }];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => ["model.ckpt_name is '', which is not one of: a", "Length is 5000, above the largest 4096 it takes"],
    refused: () => [], queueable: () => false, removedIds: () => [], offered: () => [] };
  const page = models({ dom: createDom(), pipeline: ps, api: { describeNodes: async () => ({ nodes: { L: { widgets: [] }, S: { widgets: [] } } }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  const dots = [...page.node.querySelectorAll(".cx-tile-state")];
  assert.equal(dots.length, 2);
  assert.ok(dots.every((d) => d.classList.contains("cx-tile-needs")));
});

test("a file input nothing has chosen shows “choose”, not the first file, and the one file can be picked", async () => {
  const { widgetControl } = await import("./models.js");
  const picked = [];
  const ctl = widgetControl({ name: "model_name", type: "COMBO", choices: ["only.safetensors"] }, undefined, (v) => picked.push(v));
  const select = ctl.node.querySelector("select") || ctl.node;
  assert.equal(select.value, "");
  select.value = "only.safetensors";
  select.dispatchEvent(new Event("change", { bubbles: true }));
  assert.deepEqual(picked, ["only.safetensors"]);
});

test("the simple view shows only nodes that ask for a choice (files, sampler, broken); “Show all nodes” brings back the rest", async () => {
  globalThis.localStorage = window.localStorage; localStorage.clear();
  const slots = [{ id: "m", node: "FunUIDiffusionModelLoader", group: "Loaders", inputs: { model_name: "h3.safetensors" } },
    { id: "k", node: "FunUISampler", group: "Sampling", inputs: { steps: 8 } },
    { id: "pm", node: "FunUIPromptMarkup", group: "Prompt", inputs: { text: "" } }];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [], queueable: () => true, removedIds: () => [], offered: () => [] };
  const nodes = { FunUIDiffusionModelLoader: { widgets: [{ name: "model_name", type: "COMBO", choices: ["h3.safetensors"] }] },
    FunUISampler: { widgets: [{ name: "steps", type: "INT" }] }, FunUIPromptMarkup: { widgets: [{ name: "text", type: "STRING" }] } };
  const page = models({ dom: createDom(), pipeline: ps, api: { describeNodes: async () => ({ nodes }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  const cards = () => [...page.node.querySelectorAll(".cx-tile-node .cx-tile-title")].map((n) => n.textContent).join("|");
  assert.equal(page.node.querySelectorAll(".cx-tile-node").length, 2, cards());
  assert.match(page.node.textContent, /1 more only connect features/);
  [...page.node.querySelectorAll(".cx-toggle-row")].find((r) => /Show all nodes/.test(r.textContent)).querySelector("input").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(page.node.textContent.includes("Prompt"), "the groups are back");
  page.node.remove(); localStorage.clear(); delete globalThis.localStorage;
});

test("a problem that names no slot first (a loop) still shows its nodes in the simple view, and says its words", async () => {
  const slots = [{ id: "a", node: "PlumbA", inputs: {} }, { id: "b", node: "PlumbB", inputs: {} }, { id: "c", node: "PlumbC", inputs: {} }];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => ["a feeds b feeds a: a slot cannot end up feeding itself"],
    refused: () => [], queueable: () => false, removedIds: () => [], offered: () => [] };
  const nodes = { PlumbA: { widgets: [] }, PlumbB: { widgets: [] }, PlumbC: { widgets: [] } };
  const page = models({ dom: createDom(), pipeline: ps, api: { describeNodes: async () => ({ nodes }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(page.node.querySelectorAll(".cx-tile-node").length, 2, "a and b, not c");
  assert.match(page.node.textContent, /cannot end up feeding itself/);
  page.node.remove();
});

test("a field still on screen for a node that was swapped away does not put its value on the new node", async () => {
  let slots = [{ id: "k", node: "FunUISampler", inputs: { steps: 8 } }];
  const saved = [];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    save: async (b) => { saved.push(b); }, loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [],
    queueable: () => true, removedIds: () => [], offered: () => [] };
  const nodes = { FunUISampler: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }] }, Other: { widgets: [] } };
  const page = models({ dom: createDom(), pipeline: ps, api: { describeNodes: async () => ({ nodes }), packProviders: async () => ({}) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  page.node.querySelector(".cx-tile-node").click();                    // open it
  const field = page.node.querySelector("input[type=number]");
  slots = [{ id: "k", node: "Other", inputs: {} }];                     // the swap's answer lands; the field stays
  field.value = "20";
  field.dispatchEvent(new window.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(saved, []);
  assert.match(page.node.textContent, /was swapped for Other: the value was not kept/);
  page.node.remove();
});

test("a field only some models have (SLA) is not drawn for a model known to lack the trait, nor the fields that hang off it", async () => {
  const widgets = [{ name: "model_name", type: "COMBO", choices: ["a"] }, { name: "sla", type: "BOOLEAN", requires: ["sparse_attention"] },
    { name: "sla_method", type: "COMBO", choices: ["sla"], shows: [{ input: "sla", when: [true] }] }];
  const draw = async (traits) => {
    const slots = [{ id: "model", node: "XLoader", inputs: { model_name: "a", sla: true } }];
    const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
      loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [], queueable: () => true, removedIds: () => [], offered: () => [], traits: () => traits };
    const page = models({ dom: createDom(), pipeline: ps, api: { describeNodes: async () => ({ nodes: { XLoader: { widgets } } }) } })();
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    const tile = page.node.querySelector("[class*=cx-tile]"); if (tile) (tile.closest("button") || tile).click();
    await new Promise((r) => setTimeout(r, 0));
    const text = page.node.textContent; if (process.env.DBG) console.log(JSON.stringify(text).slice(0, 600));
    page.node.remove();
    return text;
  };
  const ltx = await draw(["audio_stream", "ltx_av"]), h3 = await draw(["audio_stream", "sparse_attention"]), unknown = await draw(null);
  assert.ok(h3.includes("sla_method") && unknown.includes("sla_method"), "H3, or a model nobody recognised, keeps SLA");
  assert.ok(!ltx.includes("sla_method") && !/\bsla\b/.test(ltx.replace("sla_method", "")), "LTX has none of it");
});
