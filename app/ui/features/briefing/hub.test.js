import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import feature from "./index.js";

test("a question from the enhancer opens a window once; Send answers it, Cancel gives no answer, and the run ending closes it", async () => {
  setupDom();
  try {
    const host = document.createElement("div");
    document.body.append(host);
    let push = () => {};
    const said = [];
    const app = { api: { enhancerBriefingAnswer: (id, text) => { said.push([id, text]); return Promise.resolve(); } }, generate: { subscribe: (fn) => { push = fn; return () => {}; } } };
    const off = feature.setup({ host, app });
    push({ briefing: { id: "1", question: "Which season is it?" } });
    push({ briefing: { id: "1", question: "Which season is it?" } });                    // the same question again: still one window
    assert.equal(document.querySelectorAll("input").length, 1);
    assert.match(document.body.textContent, /Which season is it\?/);
    document.querySelector("input").value = "Winter";
    [...document.querySelectorAll("button")].find((b) => b.textContent === "Send").click();
    assert.deepEqual(said, [["1", "Winter"]]);
    push({ briefing: null });
    push({ briefing: { id: "2", question: "Day or night?" } });
    [...document.querySelectorAll("button")].find((b) => b.textContent === "Cancel").click();
    assert.deepEqual(said.at(-1), ["2", ""]);
    push({ briefing: { id: "3", question: "Indoors?" } });
    push({ briefing: null });                                                           // the run ended: the window goes (its Cancel reaches a run that is already over, which ignores it)
    assert.equal(document.querySelectorAll("input").length, 0);
    off();
  } finally { teardownDom(); }
});
