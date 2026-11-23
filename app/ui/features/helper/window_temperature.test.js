import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("the Helper writes with its own temperature (0.3 unless set), not the Enhance one", async () => {
  setupDom();
  try {
    const project = { id: "temp", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    const bodies = [];
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf", temperature: 0.9 } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {}, enhancerChatStream: (body) => { bodies.push(body); return Promise.resolve({ ok: true, reply: "ok" }); } } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    box.value = "go"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    for (let i = 0; i < 100 && !bodies.length; i++) await new Promise((r) => setTimeout(r, 10));
    assert.equal(bodies[0].temperature, 0.3);
    await new Promise((r) => setTimeout(r, 200));
  } finally { teardownDom(); }
});
