import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import feature from "./index.js";

test("the Helper button leads the row, shows only when switched on, and leaves nothing behind", async () => {
  setupDom();
  globalThis.MutationObserver = window.MutationObserver;
  try {
    const host = document.createElement("div");
    document.body.append(host);
    let on = false;
    const heard = new Set();
    const app = { project: { pref: () => on, setPref(k, v) { on = v; } }, on: (fn) => { heard.add(fn); return () => heard.delete(fn); }, editorSettings: [], hasModule: () => true, api: {} };
    const later = document.createElement("button");
    later.textContent = "Later";
    host.append(later);
    const off = feature.setup({ host, app });
    const names = () => [...host.children].map((n) => n.textContent);
    assert.deepEqual(names(), ["✦ Helper", "Later"]);
    assert.equal(host.firstElementChild.hidden, true);                    // off until the person turns it on
    on = true; heard.forEach((fn) => fn("change"));
    assert.equal(host.firstElementChild.hidden, false);
    host.prepend(document.createElement("span"));
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(host.firstElementChild.textContent, "✦ Helper");        // still first when something is added before it
    assert.equal(app.editorSettings.length, 1);
    off();
    assert.equal(app.editorSettings.length, 0);
    assert.equal(heard.size, 0);
  } finally { delete globalThis.MutationObserver; teardownDom(); }
});

test("without the prompt enhancer module the feature is hidden", async () => {
  setupDom();
  try {
    const { loadFeatures } = await import("../../shell/features.js");
    const { offer, _reset } = await import("../../shell/mounts.js");
    _reset();
    offer("timeline.status", document.createElement("div"));
    const app = { project: {}, api: {}, editorSettings: [] };
    const { mounted, hidden } = await loadFeatures(["x"], { load: async () => ({ default: feature }), app, has: (n) => n in app, modules: () => [] });
    assert.equal(mounted.length, 0);
    assert.equal(hidden.length, 1);
  } finally { teardownDom(); }
});
