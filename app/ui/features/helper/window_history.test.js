import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("text a model writes along the way is not shown as an answer, and the conversation survives a reload until Clear", async () => {
  setupDom();
  try {
    const store = new Map();
    globalThis.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
    const project = { id: "keep", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {},
        enhancerChatStream: async (body, on) => { on({ kind: "answer", text: "Okay, the user wants a longer scene, so first I should" }); return { ok: true, reply: "Scene 1 is longer now." }; } } };
    let win = openHelper(app, null);
    let modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    const button = (label) => [...modal.querySelectorAll("button")].find((b) => b.textContent === label);
    box.value = "make scene 1 longer"; box.dispatchEvent(new window.Event("input", { bubbles: true })); button("Ask helper").click();
    for (let i = 0; i < 100 && !/longer now/.test(modal.textContent); i++) await new Promise((r) => setTimeout(r, 10));
    const bubbles = [...modal.querySelectorAll(".cx-bubble")].map((b) => b.textContent);
    assert.ok(bubbles.some((t) => /Scene 1 is longer now\./.test(t)) && !bubbles.some((t) => /the user wants/.test(t)), "only the final answer is a message");
    assert.ok(store.get("funui_helper_log:keep"), "kept for a reload");
    assert.ok(button("Clear"), "has its own Clear button");
    button("Clear").click();
    assert.equal(store.has("funui_helper_log:keep"), false);
    assert.doesNotMatch(modal.textContent, /longer now/);
  } finally { delete globalThis.localStorage; teardownDom(); }
});
