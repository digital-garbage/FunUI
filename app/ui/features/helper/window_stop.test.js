import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("the Helper's Stop ends the work, and a new message while it works stops that work and goes on", async () => {
  setupDom();
  try {
    const project = { id: "p", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] }, stopped = [], pending = [], sent = [];
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {},
        enhancerChatStop: async (conversation) => { stopped.push(conversation); pending.splice(0).forEach((end) => end({ ok: false, stopped: true })); },        // the server ends the stopped request's stream
        enhancerChatStream: (body) => new Promise((resolve) => {
          sent.push(body.message);
          pending.splice(0).forEach((end) => end({ ok: false, stopped: true }));          // a new request of the conversation stops the old one
          if (body.message === "third") setTimeout(() => resolve({ ok: true, reply: "ok third" }), 20); else pending.push(resolve);
        }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    const button = (label) => [...modal.querySelectorAll("button")].find((b) => b.textContent === label);
    const type = (text) => { box.value = text; box.dispatchEvent(new window.Event("input", { bubbles: true })); };
    const wait = async (ok) => { for (let i = 0; i < 100 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    assert.ok(button("Stop").disabled, "nothing to stop yet");
    type("first"); button("Ask helper").click();
    await wait(() => pending.length === 1);
    assert.ok(!button("Stop").disabled, "Stop is there while it works");
    button("Stop").click();
    await wait(() => /Stopped\./.test(modal.textContent));
    assert.deepEqual(stopped, ["helper:p"]);
    assert.match(modal.textContent, /Stopped\./);
    type("second"); button("Ask helper").click();
    await wait(() => pending.length === 1 && sent.length === 2);
    type("third"); button("Interrupt and ask").click();                                  // new input while it works
    await wait(() => /ok third/.test(modal.textContent));
    assert.deepEqual(sent, ["first", "second", "third"]);
    assert.ok((modal.textContent.match(/Stopped\./g) || []).length >= 2, "the first and second were stopped");
    assert.match(modal.textContent, /ok third/);
  } finally { teardownDom(); }
});
