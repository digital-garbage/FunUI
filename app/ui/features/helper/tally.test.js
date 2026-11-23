import test from "node:test";
import assert from "node:assert/strict";
import { tally } from "./window.js";

test("the shortcut change lines of a turn are added up into one", () => {
  assert.deepEqual(tally(["Changed 20 shortcuts", "Changed 1 shortcut", "Changed 12 shortcuts", "Refused: x", "Changed 1 shortcut"]), ["Changed 34 shortcuts", "Refused: x"]);
  assert.deepEqual(tally(["Changed 1 shortcut"]), ["Changed 1 shortcut"]);
  assert.deepEqual(tally(["Changed the anchor", "Changed 2 shortcuts"]), ["Changed the anchor", "Changed 2 shortcuts"]);
  assert.deepEqual(tally([]), []);
});
