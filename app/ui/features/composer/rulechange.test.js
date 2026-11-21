import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { createDom } from "../../shell/dom.js";
import { chat } from "./chat.js";
// Its own file: the Allow window keeps the document it first opened in, so it cannot share a process with tests that use another one.

test("a rule change the model proposes in the chat opens the Allow window; Allow saves the new text and answers the model, Disallow saves nothing", async () => {
  setupDom();
  try {
    for (const allow of [true, false]) {
      const saved = [], decided = [], prefs = {};
      const inputs = { to_chat: true, self_edit: true, engine_model: "a/b::q.gguf", chat_prompt: "Answer in one line." };
      const app = { project: { project: { id: "p", scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { n: 0 }, selection: {}, dom: createDom(),
        pipeline: { slots: () => [{ id: "e", node: "FunUIEnhancePrompt", inputs }], subscribe: () => () => {}, save: async (x) => { saved.push(x); } },
        api: { enhancerRuns: async () => ({ runs: [] }), enhancerDefaults: async () => ({ prompts: {} }), enhancerEditDecision: async (id, yes) => { decided.push([id, yes]); },
          enhancerChatStream: async (body, on) => { on({ kind: "edit_request", id: "edit-1", field: "chat_prompt", old: "Answer in one line.", new: "Answer in two lines.", reason: "longer answers" }); await new Promise((r) => setTimeout(r, 30)); return { ok: true }; } }, on: () => () => {} };
      const page = chat(app, () => {});
      document.body.append(page.node);
      await new Promise((r) => setTimeout(r, 0));
      const box = [...page.node.querySelectorAll("textarea")].at(-2);
      box.value = "answer longer"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
      [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Send").click();
      for (let i = 0; i < 50 && !document.querySelector(".cx-modal"); i++) await new Promise((r) => setTimeout(r, 10));
      const modal = document.querySelector(".cx-modal");
      assert.ok(modal, `the Allow window opened`);
      assert.match(modal.textContent, /Allow changes to system prompt\?/);
      assert.match(modal.textContent, /Answer in two lines\./);
      [...modal.querySelectorAll("button")].find((b) => b.textContent === (allow ? "Allow" : "Disallow")).click();
      for (let i = 0; i < 50 && !decided.length; i++) await new Promise((r) => setTimeout(r, 10));
      await new Promise((r) => setTimeout(r, 50));
      assert.deepEqual(decided, [["edit-1", allow]]);
      assert.deepEqual(saved, allow ? [{ inputs: { e: { chat_prompt: "Answer in two lines." } } }] : []);
      page.node.remove();
    }
  } finally { teardownDom(); }
});
