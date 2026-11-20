import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { createDom } from "../../shell/dom.js";
import { addComment, chat, chatHook } from "./chat.js";

const slots = [{ id: "e", node: "FunUIEnhancePrompt" }, { id: "x", node: "Other" }];
const project = { editor_settings: { enhance_chat: [{ comment: "warmer", original: "a cat" }, { comment: "old", original: "a dog" }, { comment: "any" }] } };

test("comments reach the enhancer only when they are about the prompt being run, and the rest are said", () => {
  const r = chatHook({ project, scene: { text: " a cat " }, slots });
  assert.deepEqual(JSON.parse(r.inputs.e.chat).map((x) => x.comment), ["warmer", "any"]);
  assert.equal(r.notes.length, 1);
});

test("no enhancer in the pipeline, or no comments: nothing is sent", () => {
  assert.equal(chatHook({ project, scene: { text: "a cat" }, slots: [slots[1]] }), null);
  assert.equal(chatHook({ project: {}, scene: { text: "a cat" }, slots }), null);
});

test("the Chat tab shows the conversation as bubbles: your prompts on the right, the model's answers on the left, comments under their rewrite", async () => {
  setupDom();
  try {
    const prefs = { enhance_chat: [{ comment: "warmer", run: "r1" }], enhance_chat_epoch: 1 };
    const runs = [{ prompt_id: "r1", conversation: "p:1", ok: true, before: "PRE a fox POST", after: "PRE A red fox runs. POST", asked: "a fox", answered: "A red fox runs." }, { prompt_id: "r2", conversation: "p:1", ok: true, context: { used: 2048, window: 8192, compact: true }, before: "an owl", after: "An owl glides." }, { prompt_id: "r0", conversation: "p:0", ok: true, before: "old", after: "older" }];
    const app = { project: { project: { id: "p", scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { n: 0 }, selection: {}, dom: createDom(),
      pipeline: { slots: () => [{ node: "FunUIEnhancePrompt" }], subscribe: () => () => {} }, api: { enhancerRuns: async () => ({ runs }) }, on: () => () => {} };
    const page = chat(app, () => {});
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    const said = [...page.node.querySelectorAll(".cx-bubble")].map((b) => `${b.classList.contains("cx-bubble-you") ? "you" : "model"}:${b.querySelector(".cx-bubble-body").textContent}`);
    assert.match(page.node.textContent, /Context: about 2,048 of 8,192 tokens \(25%\)\. Summarised at 80%\./);
    assert.deepEqual(said, ["you:a fox", "model:A red fox runs.", "you:warmer", "you:an owl", "model:An owl glides."]);       // the older conversation (p:0) is not shown
  } finally { teardownDom(); }
});

test("the Comment button's comment is saved like one from the Chat tab, tied to the last rewrite", () => {
  const prefs = {};
  const app = { project: { project: { scenes: [{ id: "s", text: "a fox" }] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { typed: "a fox" }, selection: { focus: "s" } };
  assert.equal(addComment(app, "   ", null), false);
  assert.equal(addComment(app, " warmer ", { ok: true, prompt_id: "r1", answered: "A fox runs.", after: "PRE A fox runs." }), true);
  assert.equal(typeof prefs.enhance_chat[0].id, "string");          // each comment has its own id, so a conversation sends it once
  delete prefs.enhance_chat[0].id;
  assert.deepEqual(prefs.enhance_chat, [{ rewrites: { whole: "A fox runs." }, comment: "warmer", run: "r1", original: "a fox" }]);
});

test("in a conversation every comment goes to the enhancer, whatever prompt it was made on", () => {
  const withChat = [{ ...slots[0], inputs: { to_chat: true } }, slots[1]];
  const r = chatHook({ project, scene: { text: "a cat" }, slots: withChat });
  assert.deepEqual(JSON.parse(r.inputs.e.chat).map((x) => x.comment), ["warmer", "old", "any"]);
  assert.equal(r.notes.length, 0);
});
