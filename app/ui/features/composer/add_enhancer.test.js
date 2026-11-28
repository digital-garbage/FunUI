import { test } from "node:test";
import assert from "node:assert/strict";
import { withEnhancer } from "./add_enhancer.js";

const base = () => [
  { id: "clip", node: "FunUICLIPLoader", inputs: {} },
  { id: "positive", group: "Preparation", node: "CLIPTextEncode", roles: [{ at: "generation.prompt", input: "text", label: "Prompt" }], inputs: { clip: ["clip", 0], text: "hi" } },
  { id: "negative", node: "CLIPTextEncode", roles: [{ at: "project.negative", input: "text" }], inputs: { clip: ["clip", 0], text: "" } },
];

test("the enhancer takes the prompt and the encoder reads its answer", () => {
  const { slots } = withEnhancer(base());
  const e = slots.find((s) => s.node === "FunUIEnhancePrompt"), p = slots.find((s) => s.id === "positive");
  assert.deepEqual(e.inputs, { clip: ["clip", 0], text: "hi", enabled: false });
  assert.deepEqual(e.roles.map((r) => r.input), ["text", "enabled"]);
  assert.deepEqual(p.inputs.text, ["enhance", 0]);
  assert.equal(p.roles.length, 0);
  assert.ok(slots.indexOf(e) < slots.indexOf(p));
});

test("nothing is added twice, or where the place is not clear", () => {
  assert.ok(withEnhancer(withEnhancer(base()).slots).why);
  assert.ok(withEnhancer([]).why);
  assert.ok(withEnhancer([{ ...base()[1], inputs: { text: "" } }]).why);
});
