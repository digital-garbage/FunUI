import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("a thought opened by the person folds when its run finishes, and none is open when a new message is sent", async () => {
  setupDom();
  try {
    const project = { id: "fold", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    const runs = [];
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {},
        enhancerChatStream: (body, on) => new Promise((resolve) => { runs.push({ on, resolve }); }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    const wait = async (ok) => { for (let i = 0; i < 100 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    const ask = (text) => { box.value = text; box.dispatchEvent(new window.Event("input", { bubbles: true })); [...modal.querySelectorAll("button")].find((b) => /^(Ask helper|Interrupt and ask)$/.test(b.textContent)).click(); };
    const folds = () => [...modal.querySelectorAll(".cx-modal-body details.cx-collapsible")].filter((d) => !d.closest(".cx-stack-sticky"));
    ask("one");
    await wait(() => runs.length === 1);
    runs[0].on({ kind: "thinking", text: "first thought" });
    await wait(() => folds().length === 1);
    folds()[0].open = true;                                  // the person opens it
    await new Promise((r) => setTimeout(r, 30));
    runs[0].resolve({ ok: true, reply: "done one" });
    await wait(() => /done one/.test(modal.textContent));
    await new Promise((r) => setTimeout(r, 50));
    assert.ok(folds().every((d) => !d.open), "the finished run's thought is folded");
    folds()[0].open = true;                                  // opened again to read it
    await new Promise((r) => setTimeout(r, 30));
    ask("two");
    await wait(() => runs.length === 2);
    runs[1].on({ kind: "thinking", text: "second thought" });
    await new Promise((r) => setTimeout(r, 60));
    assert.ok(folds().length >= 1 && folds().every((d) => !d.open), "no thought is open after a new message");
    runs[1].resolve({ ok: true, reply: "done two" });
    await wait(() => /done two/.test(modal.textContent));
    await new Promise((r) => setTimeout(r, 200));
  } finally { teardownDom(); }
});
