import test from "node:test";
import assert from "node:assert/strict";
import { linked } from "./project.js";

const seconds = { sync: { with: "tail_frames", per_second: true } }, frames = { sync: { with: "tail_seconds", per_second: false } };

test("seconds and frames follow each other at the generation rate", () => {
  assert.equal(linked(seconds, 2, 25), 50);
  assert.equal(linked(seconds, 2, 24), 48);
  assert.equal(linked(frames, 48, 24), 2);
  assert.equal(linked(frames, 37, 25), 1.48);
  assert.equal(linked(seconds, 0.01, 25), 1);          // never less than one frame
  assert.equal(linked(seconds, "x", 25), null);
  assert.equal(linked({}, 2, 25), null);
  assert.equal(linked(seconds, 2, undefined), 50);          // no rate known: 25
});
