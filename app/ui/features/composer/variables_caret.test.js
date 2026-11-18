import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let variablesPanel;
test.before(async () => { setupDom(); ({ variablesPanel } = await import("./storytools.js")); });
test.after(() => teardownDom());

const make = () => {
  const project = { id: "p", variables: [{ name: "a", value: "alpha text here" }, { name: "b", value: "beta gamma" }] };
  const p = { project, setField: (k, v) => { project[k] = v; } };
  const panel = variablesPanel({ project: p, on: () => () => {}, dom: { typingIn: () => false, focusOn: (n) => n.ownerDocument.activeElement === n } }, () => {});
  window.document.body.append(panel.node);
  return panel;
};

test("a rebuilt variables list (list redraw or tab switch) puts each box's caret back", async () => {
  const first = make();
  const [a, b] = first.node.querySelectorAll("textarea");
  assert.equal(first.node.querySelectorAll(".cx-tab").length, 0, "still a plain list, no tabs");
  a.focus(); a.setSelectionRange(6, 10); a.dispatchEvent(new window.Event("select"));
  b.focus(); b.setSelectionRange(2, 2); b.dispatchEvent(new window.Event("click"));
  first.node.remove();
  const again = make();                                   // the window's tab was left and came back
  await Promise.resolve();
  const [a2, b2] = again.node.querySelectorAll("textarea");
  assert.deepEqual([a2.selectionStart, a2.selectionEnd], [6, 10]);
  assert.deepEqual([b2.selectionStart, b2.selectionEnd], [2, 2]);
  assert.equal(window.document.activeElement, b2, "the box you were in has the caret again");
});

test("switching to another browser tab and back keeps the caret of the box", async () => {
  const panel = make();
  await Promise.resolve();                                // the earlier test's saved focus lands first
  const [a] = panel.node.querySelectorAll("textarea");
  a.focus(); a.value = "x\n".repeat(60); a.setSelectionRange(20, 24); a.scrollTop = 30;
  a.dispatchEvent(new window.Event("blur"));              // the window lost focus; the box is still the active element
  await Promise.resolve();
  a.dispatchEvent(new window.Event("focus"));
  assert.equal(window.document.activeElement, a);
  assert.deepEqual([a.selectionStart, a.selectionEnd], [20, 24]);
});
