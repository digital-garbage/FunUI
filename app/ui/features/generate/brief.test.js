import test from "node:test";
import assert from "node:assert/strict";
import { brief } from "./brief.js";

test("a short error stays as it is and a long one keeps its start and its end", () => {
  assert.equal(brief("it broke"), "it broke");
  const long = `Generation failed in FunUISampler: ${"#define X 1 ".repeat(400)}nvrtc: error: failed to open libnvrtc-builtins.so.13.0.`;
  const out = brief(long);
  assert.ok(out.length < 450 && out.startsWith("Generation failed in FunUISampler") && out.endsWith("libnvrtc-builtins.so.13.0."));
  assert.equal(brief(""), "");
});
