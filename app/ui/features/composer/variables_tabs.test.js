import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let variablesPanel;
test.before(async () => { setupDom(); ({ variablesPanel } = await import("./storytools.js")); });
test.after(() => teardownDom());

test("each variable tab keeps its caret and scroll when you switch away and back", () => {
  const project = { id: "p", variables: [{ name: "a", value: "alpha text here" }, { name: "b", value: "beta" }] };
  const p = { project, setField: (k, v) => { project[k] = v; } };
  const panel = variablesPanel({ project: p, on: () => () => {}, dom: { typingIn: () => false } }, () => {});
  document.body.append(panel.node);
  const tabs = () => [...panel.node.querySelectorAll(".cx-tab")];
  const box = () => panel.node.querySelector("textarea");
  assert.equal(tabs().length, 2);
  box().focus(); box().setSelectionRange(6, 10); box().scrollTop = 7;
  tabs()[1].click();
  assert.equal(box().value, "beta");
  box().setSelectionRange(1, 2);
  tabs()[0].click();
  assert.equal(box().value, "alpha text here");
  assert.deepEqual([box().selectionStart, box().selectionEnd], [6, 10], "caret restored");
  tabs()[1].click();
  assert.deepEqual([box().selectionStart, box().selectionEnd], [1, 2], "the other tab kept its own");
});
