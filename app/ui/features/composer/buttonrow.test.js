import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import feature from "./index.js";

test("the row reads Helper first, then the others, then Sampler, then Composer last, even when another feature adds a button later", async () => {
  setupDom();
  try {
    const host = document.createElement("div");
    document.body.append(host);
    const app = { project: { pref: () => false, setPref() {} }, on: () => () => {}, editorSettings: [], inputHooks: [], hasModule: () => true, api: {}, generate: null };
    const off = feature.setup({ host, app });
    const names = () => [...host.children].map((n) => n.textContent);
    assert.deepEqual(names(), ["✦ Helper", "◆ Composer"]);
    const later = document.createElement("button"), sampler = document.createElement("button");
    later.textContent = "Later"; sampler.textContent = "⏱ Sampler"; sampler.dataset.row = "sampler";
    host.append(sampler, later);
    await new Promise((r) => setTimeout(r, 20));
    assert.deepEqual(names(), ["✦ Helper", "Later", "⏱ Sampler", "◆ Composer"]);
    off();
  } finally { teardownDom(); }
});
