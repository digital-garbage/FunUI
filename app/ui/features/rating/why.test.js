import test from "node:test";
import assert from "node:assert/strict";

test("a dislike asks what was wrong when that is switched on and the enhancer is on; a like, or the switch off, asks nothing", async () => {
  const { setupDom, teardownDom } = await import("../../composer/tests/_dom.js");
  setupDom();
  try {
    const { default: rating } = await import("./index.js");
    const flush = () => new Promise((r) => setTimeout(r, 20));
    const run = async (ask, enabled, choice) => {
      const project = { scenes: [{ id: "a", text: "a dog", source: { type: "t2v" } }], scene_renders: { a: { media: { filename: "a.mp4" }, promptId: "p1", text: "a dog" } }, editor_settings: {} };
      const prefs = { enhance_ask_dislike: ask };
      const app = { project: { project, selected: null, pref: (k, d) => (k in prefs ? prefs[k] : d), setPref: (k, v) => { prefs[k] = v; }, edit: (fn) => fn(project) },
        selection: { focus: "a", pick() {} }, api: { rateTaste: async () => ({}), enhancerRuns: async () => ({ runs: [] }) },
        pipeline: { slots: () => [{ node: "FunUIEnhancePrompt", id: "e", inputs: { enabled } }] }, lastRun: { n: 0 }, on: () => () => {} };
      const host = document.createElement("div");
      document.body.append(host);
      rating.setup({ host, app });
      const rateBtn = [...host.querySelectorAll("button")].find((b) => /Rate scene/.test(b.textContent));
      rateBtn.click();
      await flush();
      [...document.querySelectorAll("button, .cx-choice, li, [role=button]")].find((n) => n.textContent.trim().startsWith(choice))?.click();
      await flush();
      const asked = document.body.textContent.includes("What was wrong with it?");
      document.body.innerHTML = "";
      return asked;
    };
    assert.equal(await run(true, true, "Disliked"), true);
    assert.equal(await run(false, true, "Disliked"), false);
    assert.equal(await run(true, false, "Disliked"), false);
    assert.equal(await run(true, true, "Liked"), false);
  } finally { teardownDom(); }
});
