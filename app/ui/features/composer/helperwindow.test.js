import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./helper.js";
// Its own file: a modal keeps the document it first opened in.

test("the Helper window opens in the middle, sends the project sheet, asks Allow for a change and applies it", async () => {
  setupDom();
  try {
    const project = { id: "p", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] }, sent = [], decided = [];
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async (id, yes) => { decided.push([id, yes]); },
        enhancerChatStream: async (body, on) => { sent.push(body); on({ kind: "thinking", text: "scene 1 needs a sitting dog" }); on({ kind: "action_request", id: "act-1", tool: "edit_scene", args: { scene: 1, text: "a dog sits" }, summary: "Scene 1: change the prompt to: a dog sits", reason: "asked" }); await new Promise((r) => setTimeout(r, 80)); return { ok: true, reply: "Done." }; } } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal");
    assert.ok(modal, "the window opened");
    const box = modal.querySelector("textarea");
    box.value = "make the dog sit"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    for (let i = 0; i < 50 && document.querySelectorAll(".cx-modal").length < 2; i++) await new Promise((r) => setTimeout(r, 10));
    const card = [...document.querySelectorAll(".cx-modal")].pop();
    assert.match(modal.textContent, /scene 1 needs a sitting dog/);       // the helper's thinking shows in its own window
    assert.match(card.textContent, /a dog sits/);
    [...card.querySelectorAll("button")].find((b) => b.textContent === "Allow").click();
    for (let i = 0; i < 50 && !decided.length; i++) await new Promise((r) => setTimeout(r, 10));
    assert.deepEqual(decided, [["act-1", true]]);
    assert.equal(project.scenes[0].text, "a dog sits");
    assert.equal(sent[0].conversation, "helper:p");
    for (let i = 0; i < 50 && !modal.textContent.includes("Done."); i++) await new Promise((r) => setTimeout(r, 10));
    assert.match(modal.textContent, /Done\./);
    assert.match(sent[0].helper.sheet, /Scene 1 .*a dog runs/);
  } finally { teardownDom(); }
});
