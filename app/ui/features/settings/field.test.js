import test from "node:test";
import assert from "node:assert/strict";
import { clampBlocks } from "./field.js";

test("a block list is held to the model's blocks", () => {
  assert.equal(clampBlocks("2-10,29-46", 48), "2-10,29-46");
  assert.equal(clampBlocks("2-10,29-46", 40), "2-10,29-39");
  assert.equal(clampBlocks("43-44", 40), "");
  assert.equal(clampBlocks("5,50,60-70,38-45", 40), "5,38-39");
  assert.equal(clampBlocks("30-20", 25), "20-24");
});

test("nothing is changed when the count is unknown, and odd entries are left for the module to say", () => {
  assert.equal(clampBlocks("2-99", null), "2-99");
  assert.equal(clampBlocks("2-99", 0), "2-99");
  assert.equal(clampBlocks("x, 3", 10), "x,3");
});
