import test from "node:test";
import assert from "node:assert/strict";
import hub from "./index.js";
import { styleFor } from "./intent.js";

test("words choose the style", () => {
  assert.equal(styleFor(""), "natural");
  assert.equal(styleFor("a woman walks"), "natural");
  assert.equal(styleFor("a fast chase, running"), "decelerate");
  assert.equal(styleFor("a calm, still portrait"), "freeze");
  assert.equal(styleFor("a slow pan"), "accelerate");
  assert.equal(styleFor("a seamless loop of waves"), "loop");
});

const make = (values, off = false) => { const app = { pipeline: { currentValues: () => values, allOff: () => off }, inputHooks: [] }; hub.setup({ app }); return app.inputHooks[0]; };
const slots = [{ id: "5", node: "FunUIModifierSettings" }];

test("auto becomes the shot's own style in that run's settings, other values kept", async () => {
  const hook = make({ temporal_style: { style: "auto" }, other: { a: 1 } });
  const out = await hook({ scene: { text: "calm still portrait" }, slots });
  assert.deepEqual(JSON.parse(out.inputs[5].settings), { temporal_style: { style: "freeze" }, other: { a: 1 } });
});

test("nothing happens for a fixed style, with everything off, or without a settings node", async () => {
  assert.deepEqual(await make({ temporal_style: { style: "pulse" } })({ scene: { text: "x" }, slots }), {});
  assert.deepEqual(await make({ temporal_style: { style: "auto" } }, true)({ scene: { text: "x" }, slots }), {});
  assert.deepEqual(await make({ temporal_style: { style: "auto" } })({ scene: { text: "x" }, slots: [] }), {});
});
