import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("after a refresh the window follows a request that is still running on the server", async () => {
  let more = null, finish = null, finishing = null;
  setupDom();
  try {
    const store = new Map(), asked = [];
    globalThis.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
    const project = { id: "back", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {},
        enhancerChatAttach: (conversation, on) => new Promise((resolve) => {
          asked.push(conversation);
          on({ kind: "begin", message: "rename everything", seconds: 12 }); on({ kind: "thinking", text: "working it out " }); on({ kind: "status", text: "Checklist: 2 of 5 done" });
          more = on; finish = resolve; finishing = resolve;
        }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal");
    const wait = async (ok) => { for (let i = 0; i < 200 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    await wait(() => /Checklist: 2 of 5 done/.test(modal.textContent));
    assert.deepEqual(asked, ["helper:back"]);
    assert.match(modal.textContent, /rename everything/, "the request it was asked is shown");
    assert.match(modal.textContent, /Checklist: 2 of 5 done/, "and where it has got to");
    assert.ok(![...modal.querySelectorAll("button")].find((b) => b.textContent === "Stop").disabled, "Stop is there while it runs");
    more({ kind: "thinking", text: "and further" });
    finish({ ok: true, reply: "Renamed all five." });
    await wait(() => /Renamed all five\./.test(modal.textContent));
    assert.match(modal.textContent, /Renamed all five\./);
    assert.ok(store.get("funui_helper_log:back"), "the finished turn is kept like any other");
    await new Promise((r) => setTimeout(r, 200));
  } finally { if (finishing) finishing({ ok: false, status: "test over" }); delete globalThis.localStorage; teardownDom(); }
});
