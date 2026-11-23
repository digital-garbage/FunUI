import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { openHelper } from "./window.js";
// Its own file: a modal keeps the document it first opened in.

test("the window does not redraw between a press and its release, so the thinking line can be opened while an answer streams in", async () => {
  setupDom();
  try {
    const project = { id: "hold", frame_rate: 24, num_frames_per_scene: 97, scenes: [{ id: "a", text: "a dog runs" }] };
    let push = null, finish = null;
    const app = { project: { project, pref: () => undefined, edit: (fn) => fn(project) }, pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: { engine_model: "a/b::q.gguf" } }] },
      api: { enhancerEditDecision: async () => {}, enhancerChatStop: async () => {},
        enhancerChatStream: (body, on) => new Promise((resolve) => { push = on; finish = resolve; }) } };
    openHelper(app, null);
    const modal = document.querySelector(".cx-modal"), box = modal.querySelector("textarea");
    box.value = "go"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Ask helper").click();
    const wait = async (ok) => { for (let i = 0; i < 100 && !ok(); i++) await new Promise((r) => setTimeout(r, 10)); };
    await wait(() => push);
    push({ kind: "thinking", text: "first thought" });
    const line = () => modal.querySelector("details, .cx-collapsible");
    await wait(() => line());
    const before = line();
    assert.ok(before, "the thinking line is there");
    const history = before.parentNode;
    history.dispatchEvent(new window.Event("pointerdown", { bubbles: true }));
    push({ kind: "thinking", text: " and more" });                       // a redraw's worth of news arrives while the button is down
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(line(), before, "the pressed thing is still the same element");
    history.dispatchEvent(new window.Event("pointerup", { bubbles: true }));
    await wait(() => line() !== before);
    assert.match(modal.textContent, /first thought and more/, "the news is shown once the pointer is up");
    finish({ ok: true, reply: "all done" });
    await wait(() => /all done/.test(modal.textContent));
    await new Promise((r) => setTimeout(r, 50));            // the clock beside the thinking stops with the turn
  } finally { teardownDom(); }
});
