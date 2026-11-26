import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("a Helper window opened again while the helper still works shows it working, with Stop, and shows its end", async () => {
  setupDom();
  try {
    const project = { id: "reopen", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] }, pending = [];
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {}, enhancerChatStream: () => new Promise((resolve) => { pending.push(resolve); }) } };
    const wait = async (ok) => { for (let i = 0; i < 100 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    const first = openHelper(app, null);
    let modal = document.querySelector(".cx-modal");
    const box = modal.querySelector("textarea");
    box.value = "go"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    await wait(() => pending.length === 1);
    first.close("manual");
    openHelper(app, null);
    modal = document.querySelector(".cx-modal");
    const button = (label) => [...modal.querySelectorAll("button")].find((b) => b.textContent === label);
    assert.ok(!button("Stop").disabled, "Stop works in the window opened again");
    pending[0]({ ok: true, reply: "all done" });
    await wait(() => /all done/.test(modal.textContent));
    assert.match(modal.textContent, /all done/);
    assert.ok(button("Stop").disabled, "nothing left to stop");
  } finally { teardownDom(); }
});
