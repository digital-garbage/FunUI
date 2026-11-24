import test from "node:test";
import assert from "node:assert/strict";
import { pickFiles } from "./repo_pick.js";

const all = ["imatrix_unsloth.gguf", "m-Q4_K_M.gguf", "m-Q8_0.gguf", "mmproj-F16.gguf", "big-00001-of-00002.gguf", "big-00002-of-00002.gguf"];

test("a new model file replaces the old one, keeps the projector and puts the model first", () => {
  assert.deepEqual(pickFiles(all, ["m-Q4_K_M.gguf", "mmproj-F16.gguf"], "m-Q8_0.gguf", true), ["m-Q8_0.gguf", "mmproj-F16.gguf"]);
  assert.deepEqual(pickFiles(all, ["m-Q4_K_M.gguf"], "mmproj-F16.gguf", true), ["m-Q4_K_M.gguf", "mmproj-F16.gguf"]);
  assert.deepEqual(pickFiles(all, ["mmproj-F16.gguf"], "m-Q4_K_M.gguf", true), ["m-Q4_K_M.gguf", "mmproj-F16.gguf"]);
});

test("the parts of a split model go together and an importance matrix is never picked", () => {
  assert.deepEqual(pickFiles(all, ["big-00001-of-00002.gguf"], "big-00002-of-00002.gguf", true), ["big-00001-of-00002.gguf", "big-00002-of-00002.gguf"]);
  assert.deepEqual(pickFiles(all, ["m-Q4_K_M.gguf"], "imatrix_unsloth.gguf", true), ["m-Q4_K_M.gguf"]);
  assert.deepEqual(pickFiles(all, ["m-Q4_K_M.gguf", "mmproj-F16.gguf"], "m-Q4_K_M.gguf", false), ["mmproj-F16.gguf"]);
});
