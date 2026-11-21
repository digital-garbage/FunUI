import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { createDom } from "../../shell/dom.js";
import { addComment, applyNotes, chat, chatHook, learnClock } from "./chat.js";

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
    const runs = [{ prompt_id: "r1", conversation: "p:1", ok: true, before: "PRE a fox POST", after: "PRE A red fox runs. POST", asked: "a fox", answered: "A red fox runs.", thinking: "The fox should be red." }, { prompt_id: "r2", conversation: "p:1", ok: true, context: { used: 2048, window: 8192, compact: true }, compacted: true, summary: "A fox story in the snow.", before: "an owl", after: "An owl glides." }, { prompt_id: "r0", conversation: "p:0", ok: true, before: "old", after: "older" }];
    const app = { project: { project: { id: "p", scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { n: 0 }, selection: {}, dom: createDom(),
      pipeline: { slots: () => [{ node: "FunUIEnhancePrompt" }], subscribe: () => () => {} }, api: { enhancerRuns: async () => ({ runs }) }, on: () => () => {} };
    const page = chat(app, () => {});
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    const said = [...page.node.querySelectorAll(".cx-bubble")].map((b) => `${b.classList.contains("cx-bubble-you") ? "you" : "model"}:${b.querySelector(".cx-bubble-body").textContent}`);
    assert.match(page.node.textContent, /Context \(last rewrite\): about 2,048 of 8,192 tokens \(25%\)\. Summarised at 80%\./);
    assert.match(page.node.textContent, /summarised here[\s\S]*Summary the model now works from[\s\S]*A fox story in the snow\./);
    assert.deepEqual(said, ["you:a fox", "model:The fox should be red.", "model:A red fox runs.", "you:warmer", "you:an owl", "model:An owl glides."]);       // the older conversation (p:0) is not shown
  } finally { teardownDom(); }
});

test("comments sit in the order of their chat ids among the messages, even when their run is gone", async () => {
  setupDom();
  try {
    const prefs = { enhance_chat: [{ comment: "late", chat_id: 400 }, { comment: "first", chat_id: 50 }, { comment: "middle", chat_id: 250, run: "gone" }], enhance_chat_epoch: 1 };
    const runs = [{ prompt_id: "r2", conversation: "p:1", ok: true, chat_id: 300, asked: "an owl", answered: "An owl glides.", before: "an owl", after: "An owl glides." }, { prompt_id: "r1", conversation: "p:1", ok: true, chat_id: 100, asked: "a fox", answered: "A red fox runs.", before: "a fox", after: "A red fox runs." }];
    const app = { project: { project: { id: "p", scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { n: 0 }, selection: {}, dom: createDom(),
      pipeline: { slots: () => [{ node: "FunUIEnhancePrompt" }], subscribe: () => () => {} }, api: { enhancerRuns: async () => ({ runs }) }, on: () => () => {} };
    const page = chat(app, () => {});
    document.body.append(page.node);
    await new Promise((r) => setTimeout(r, 0));
    const said = [...page.node.querySelectorAll(".cx-bubble")].map((b) => b.querySelector(".cx-bubble-body").textContent);
    assert.deepEqual(said, ["first", "a fox", "A red fox runs.", "middle", "an owl", "An owl glides.", "late"]);
  } finally { teardownDom(); }
});

test("the Comment button's comment is saved like one from the Chat tab, tied to the last rewrite", () => {
  const prefs = {};
  const app = { project: { project: { scenes: [{ id: "s", text: "a fox" }] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { typed: "a fox" }, selection: { focus: "s" } };
  assert.equal(addComment(app, "   ", null), false);
  assert.equal(addComment(app, " warmer ", { ok: true, prompt_id: "r1", answered: "A fox runs.", after: "PRE A fox runs." }), true);
  assert.equal(typeof prefs.enhance_chat[0].id, "string");          // each comment has its own id, so a conversation sends it once
  assert.ok(prefs.enhance_chat[0].chat_id > 0);          // and a chat id that only rises
  delete prefs.enhance_chat[0].id; delete prefs.enhance_chat[0].chat_id;
  assert.deepEqual(prefs.enhance_chat, [{ rewrites: { whole: "A fox runs." }, comment: "warmer", run: "r1", original: "a fox" }]);
});

test("a comment takes the next chat id after everything it has seen, on the enhancer's clock, so it keeps its place", () => {
  const base = Date.now() + 10_000_000, prefs = { enhance_chat: [{ comment: "old", chat_id: base - 1000 }] };
  const app = { project: { project: { scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: {}, selection: {} };
  learnClock({ now: Date.now() });                       // the enhancer's clock is far behind the numbers it has already given out: ids still go up
  addComment(app, "new", { ok: true, prompt_id: "r9", chat_id: base, after: "x" });
  assert.equal(prefs.enhance_chat[1].chat_id, base + 1);
  learnClock({ now: Date.now() + 50_000 });              // a clock ahead: the comment follows that clock
  addComment(app, "newer", null);
  assert.ok(prefs.enhance_chat[2].chat_id > base + 1);
});

test("in a conversation every comment goes to the enhancer, whatever prompt it was made on", () => {
  const withChat = [{ ...slots[0], inputs: { to_chat: true } }, slots[1]];
  const r = chatHook({ project, scene: { text: "a cat" }, slots: withChat });
  assert.deepEqual(JSON.parse(r.inputs.e.chat).map((x) => x.comment), ["warmer", "old", "any"]);
  assert.equal(r.notes.length, 0);
});

test("Send asks the model now (a conversation and an encoder of its own are needed); Comment only keeps the note for the next prompt", async () => {
  setupDom();
  try {
    const prefs = {}, asked = [];
    const mk = (inputs) => ({ project: { project: { id: "p", scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { n: 0 }, selection: {}, dom: createDom(),
      pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs }], subscribe: () => () => {} }, api: { enhancerRuns: async () => ({ runs: [] }), enhancerChat: async (b) => { asked.push(b); return { ok: true }; } }, on: () => () => {} });
    const open = async (inputs) => { const page = chat(mk(inputs), () => {}); document.body.append(page.node); await new Promise((r) => setTimeout(r, 0)); return page; };
    const button = (page, label) => [...page.node.querySelectorAll("button")].find((b) => b.textContent === label);
    const off = await open({ to_chat: true });                                       // no encoder of its own: nothing to answer with
    assert.equal(button(off, "Send").disabled, true);
    const page = await open({ to_chat: true, encoder_file: "g.safetensors", temperature: 0.5 });
    const box = [...page.node.querySelectorAll("textarea")].pop();        // the Message box comes after the Notes box
    box.value = "make it warmer?"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    button(page, "Send").click();
    await new Promise((r) => setTimeout(r, 0));
    assert.deepEqual(asked.map((a) => [a.conversation, a.message, a.encoder_file, a.temperature]), [["p:0", "make it warmer?", "g.safetensors", 0.5]]);
    assert.equal(prefs.enhance_chat, undefined);                                    // a message is not a comment
    const before = asked.length;
    button(page, "Refresh").click();                                                // reloads the runs; asks the model nothing
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(asked.length, before);
    box.value = "slower"; box.dispatchEvent(new window.Event("input", { bubbles: true }));
    button(page, "Comment").click();
    assert.equal(prefs.enhance_chat.length, 1);
    assert.equal(asked.length, 1);                                                  // and a comment asks nothing
  } finally { teardownDom(); }
});

test("notes: the model's REMEMBER and FORGET lines edit the list, the notes go to every run, and an empty list sends nothing", () => {
  assert.equal(applyNotes("no rain\nwide shots", ["warm light", "Wide shots"], ["1"]), "wide shots\nwarm light");   // 1 dropped, a repeat not added twice
  assert.equal(applyNotes("a\nb", [], ["B", "#1"]), "");                                                         // by text, or by number with a hash
  const r = chatHook({ project: { editor_settings: { enhance_notes: " warm light " } }, scene: { text: "x" }, slots: [{ id: "e", node: "FunUIEnhancePrompt" }] });
  assert.deepEqual(r.inputs, { e: { notes: "warm light" } });
  assert.equal(chatHook({ project: { editor_settings: { enhance_notes: "  " } }, scene: { text: "x" }, slots: [{ id: "e", node: "FunUIEnhancePrompt" }] }), null);
});

test("the Chat tab opens at the bottom of its scrolling window, and Top ^ goes back to the start", async () => {
  setupDom();
  try {
    const frame = document.createElement("div");
    frame.style.overflowY = "auto";
    let top = 40;
    Object.defineProperty(frame, "scrollHeight", { value: 900 });
    Object.defineProperty(frame, "scrollTop", { get: () => top, set: (v) => { top = v; } });
    document.body.append(frame);
    const prefs = {};
    const app = { project: { project: { id: "p", scenes: [] }, pref: (k, d) => prefs[k] ?? d, setPref: (k, v) => { prefs[k] = v; } }, lastRun: { n: 0 }, selection: {}, dom: createDom(),
      pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", inputs: {} }], subscribe: () => () => {} }, api: { enhancerRuns: async () => ({ runs: [] }) }, on: () => () => {} };
    const page = chat(app, () => {});
    frame.append(page.node);
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(top, 900);                                                         // opened at the end
    [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Top ^").click();
    assert.equal(top, 0);
  } finally { teardownDom(); }
});

test("the rules the model may edit are the person's own text, else the built-in one", async () => {
  const { rulesOf } = await import("./chat.js");
  const app = { api: { enhancerDefaults: async () => ({ instructions: "Rewrite.", styles: { dynamic: "Dynamic rewrite." }, reference_intro: "Refs:", prompts: { chat_prompt: "Chat.", notes_prompt: "Notes." } }) } };
  const rules = await rulesOf(app, { style: "dynamic", chat_prompt: "My chat rule." });
  assert.equal(rules.instructions, "Dynamic rewrite.");
  assert.equal(rules.chat_prompt, "My chat rule.");
  assert.equal(rules.notes_prompt, "Notes.");
  assert.equal(rules.reference_intro, "Refs:");
  assert.ok(rules.rating_prefix);
});

test("a proposed change to a system prompt opens a diff and waits for Allow or Disallow", async () => {
  setupDom();
  try {
    const { askEdit } = await import("./editrequest.js");
    for (const [label, want] of [["Allow", true], ["Disallow", false]]) {
      const answer = askEdit({ field: "chat_prompt", old: "Be kind. Answer in one line.", new: "Be kind. Answer in two lines.", reason: "the user wants longer answers" });
      const modal = document.querySelector(".cx-modal");
      assert.match(modal.textContent, /Allow changes to system prompt\?/);
      assert.match(modal.textContent, /Chat answers/);
      assert.match(modal.textContent, /- Answer in one line\./);
      assert.match(modal.textContent, /\+ Answer in two lines\./);
      assert.equal(modal.querySelectorAll(".cx-code-add").length, 1);
      assert.equal(modal.querySelectorAll(".cx-code-del").length, 1);
      [...modal.querySelectorAll("button")].find((b) => b.textContent === label).click();
      assert.equal(await answer, want);
      assert.equal(document.querySelector(".cx-modal"), null);
    }
  } finally { teardownDom(); }
});
