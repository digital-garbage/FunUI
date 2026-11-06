import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../composer/tests/_dom.js";

test.before(() => setupDom());
const tick = (ms = 30) => new Promise((r) => setTimeout(r, ms));

test("Restart ComfyUI says it is restarting once, not twice", async () => {
  const { createMaintenance } = await import("./maintenance.js");
  // The server answers that it is restarting, then stays down.
  const api = { git: async () => ({ ok: true, restarting: true }), health: async () => { throw new Error("down"); } };
  const m = createMaintenance({ api, flush: async () => {} });
  m.restart();
  await tick();
  [...document.querySelectorAll("button")].find((b) => /Go ahead/.test(b.textContent)).click();
  await tick(100);
  const said = (document.body.textContent.match(/Restarting ComfyUI…/g) || []).length;
  assert.equal(said, 1, "the working card repeats the restarting line");
});
test("Restart ComfyUI says which modules it will install on the way up", async () => {
  const { createMaintenance } = await import("./maintenance.js");
  const api = { addons: async () => [{ title: "Sharp Hands", state: "pending" }, { title: "Other", state: "installed" }], git: async () => ({ ok: true, restarting: true }), health: async () => { throw new Error("down"); } };
  createMaintenance({ api, flush: async () => {} }).restart();
  await tick(100);
  const text = document.body.textContent;
  [...document.querySelectorAll("button")].find((b) => /Go ahead/.test(b.textContent))?.click();
  assert.match(text, /download and install: Sharp Hands/);
  assert.doesNotMatch(text, /Other/);
});
// The restart card polls the stopped server forever by design, so the process is ended when the test is done.
test.after(() => { teardownDom(); setTimeout(() => process.exit(0), 10); });
