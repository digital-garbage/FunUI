import test, { mock } from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let story;
test.before(async () => { setupDom(); ({ story } = await import("./index.js")); });
test.after(() => teardownDom());

test("text typed in one project is never split into another one opened before the pause ran out", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  const splits = [], edits = [];
  const a = { id: "A", scenes: [{ id: "a1", text: "old" }] }, b = { id: "B", scenes: [{ id: "b1", text: "one" }, { id: "b2", text: "two" }] };
  const heard = new Set();
  const p = { project: a, get scenes() { return this.project.scenes; }, edit: (fn) => { edits.push(this.project?.id); return fn(this.project); }, flush: async () => {} };
  const api = { storyMarkers: async () => ({ markers: ["qcut"] }), storySplit: async (text) => { splits.push(text); return { scenes: [] }; } };
  const app = { project: p, api, on: (f) => { heard.add(f); return () => heard.delete(f); }, dom: { focusOn: () => true, typingIn: () => false }, beforeRun: [] };
  const node = story(app, () => {});
  const area = (node.node || node).querySelector("textarea");
  area.value = "typed in A";
  area.dispatchEvent(new window.Event("input", { bubbles: true }));
  p.project = b;                                      // a keyboard-only project switch: no blur, so nothing committed the text
  heard.forEach((f) => f());
  mock.timers.tick(800);
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(splits, [], "A's text was not sent to split for B");
  assert.deepEqual(edits, [], "B's scenes were not rewritten");
  mock.timers.reset();
});
