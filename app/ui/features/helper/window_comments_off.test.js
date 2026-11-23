import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

async function run(on) {
  setupDom();
  try {
    const project = { id: on ? "cm-on" : "cm-off", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    let push = null, finish = null;
    const app = { project: { project, pref: (k) => (k === "helper_comments" ? on : undefined), edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {}, enhancerChatStream: (body, ev) => new Promise((resolve) => { push = ev; finish = resolve; }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    box.value = "go"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    const wait = async (ok) => { for (let i = 0; i < 100 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    await wait(() => push);
    push({ kind: "answer", text: "Sorry, let me look again." });
    push({ kind: "doing", tool: "find_shortcuts" });
    await new Promise((r) => setTimeout(r, 150));
    const outside = modal.querySelector(".cx-modal-body").textContent;
    finish({ ok: true, reply: "all done" });
    await wait(() => /all done/.test(modal.textContent));
    await new Promise((r) => setTimeout(r, 200));
    return /let me look again/.test([...modal.querySelectorAll(".cx-hint")].map((h) => h.textContent).join("|")) || (!modal.querySelector("details") ? /let me look again/.test(outside) : false);
  } finally { teardownDom(); }
}

test("a comment written before an action is stays in the thinking line when the switch is off", async () => {
  assert.equal(await run(false), false);
});
