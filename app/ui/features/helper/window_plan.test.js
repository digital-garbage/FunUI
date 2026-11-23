import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("the checklist is pinned and updated, the working label names the tool at work, and a dropped stream is followed again", async () => {
  let finishing = null;
  setupDom();
  try {
    const project = { id: "plan", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    let attaches = 0, first = null;
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {},
        enhancerChatStream: (body, on) => new Promise((resolve, reject) => {
          first = on;
          on({ kind: "plan", items: [{ text: "rename a", state: "open", note: "" }, { text: "rename b", state: "open", note: "" }] });
          on({ kind: "doing", tool: "edit_shortcuts" });
          setTimeout(() => reject(new Error("The answer stopped before it finished.")), 150);        // the connection drops
        }),
        enhancerChatAttach: (conversation, on) => !first ? Promise.resolve(null) : new Promise((resolve) => {          // nothing runs when the window opens
          attaches += 1; finishing = resolve;
          on({ kind: "begin", message: "go", seconds: 3 });
          on({ kind: "plan", items: [{ text: "rename a", state: "done", note: "done" }, { text: "rename b", state: "open", note: "" }] });
          on({ kind: "doing", tool: "read_prompt" });
          setTimeout(() => resolve({ ok: true, reply: "Both renamed." }), 250);
        }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    const wait = async (ok) => { for (let i = 0; i < 300 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    box.value = "go"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    await wait(() => /Checklist: 0 of 2 done/.test(modal.textContent));
    assert.match(modal.textContent, /○ 1\. rename a/, "the checklist is shown");
    assert.match(modal.textContent, /Editing shortcuts… \d+s/, "the label names what is being done");
    await wait(() => attaches === 1);
    await wait(() => /Checklist: 1 of 2 done/.test(modal.textContent));
    assert.match(modal.textContent, /✓ 1\. rename a/, "the list moved on");
    assert.match(modal.textContent, /Reading prompts… \d+s/);
    await wait(() => /Both renamed\./.test(modal.textContent));
    assert.match(modal.textContent, /Both renamed\./, "the dropped stream did not end the turn");
    assert.doesNotMatch(modal.textContent, /No reply/);
    await new Promise((r) => setTimeout(r, 200));
  } finally { if (finishing) finishing({ ok: false }); teardownDom(); }
});
