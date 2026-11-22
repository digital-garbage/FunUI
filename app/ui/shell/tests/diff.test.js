import test from "node:test";
import assert from "node:assert/strict";
import { lineDiff, pieces } from "../diff.js";

test("a changed sentence shows as one removed and one added line, the rest stays", () => {
  const before = "Be kind. Answer in one line. Never rhyme.", after = "Be kind. Answer in two lines. Never rhyme.";
  assert.deepEqual(lineDiff(before, after), [{ t: " ", text: "Be kind." }, { t: "-", text: "Answer in one line." }, { t: "+", text: "Answer in two lines." }, { t: " ", text: "Never rhyme." }]);
});

test("an added rule is only additions and a removed one only removals", () => {
  assert.deepEqual(lineDiff("A.\nB.", "A.\nB.\nC.").filter((r) => r.t !== " "), [{ t: "+", text: "C." }]);
  assert.deepEqual(lineDiff("A.\nB.", "A.").filter((r) => r.t !== " "), [{ t: "-", text: "B." }]);
  assert.equal(lineDiff("Same.", "Same.").every((r) => r.t === " "), true);
  assert.deepEqual(pieces(""), [""]);
});
