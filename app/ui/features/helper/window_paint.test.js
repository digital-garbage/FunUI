import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("a stream of news does not make the window redraw its history for every piece", async () => {
  setupDom();
  try {
    const project = { id: "paint", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    let push = null, finish = null;
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {}, enhancerChatStream: (body, on) => new Promise((resolve) => { push = on; finish = resolve; }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    box.value = "go"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    const wait = async (ok) => { for (let i = 0; i < 200 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    await wait(() => push);
    const history = modal.querySelector(".cx-bubble").parentNode;
    let redraws = 0;
    const seen = new window.MutationObserver((list) => { redraws += list.filter((m) => m.type === "childList" && m.target === history).length; });
    seen.observe(history, { childList: true });
    for (let i = 0; i < 100; i++) { push({ kind: "thinking", text: `t${i} ` }); await new Promise((r) => setTimeout(r, 2)); }
    await wait(() => /t99/.test(modal.textContent));
    seen.disconnect();
    assert.match(modal.textContent, /t99/, "the last piece is shown");
    assert.ok(redraws > 0 && redraws < 25, `redraws: ${redraws}`);
    finish({ ok: true, reply: "all done" });
    await wait(() => /all done/.test(modal.textContent));
    await new Promise((r) => setTimeout(r, 50));
  } finally { teardownDom(); }
});
