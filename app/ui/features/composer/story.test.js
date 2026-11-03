import test from "node:test";
import assert from "node:assert/strict";
import { applyStory, clash, joinStory, match, shownStory, storyRoots } from "./story.js";

const proj = () => ({ scenes: [{ id: "a", text: "one" }, { id: "b", text: "two" }, { id: "v", text: "", source: { type: "video", media_ref: "m" } }, { id: "c", text: "three" }], scene_renders: {}, scene_ghosts: [] });
const texts = (p) => p.scenes.map((s) => `${s.id}:${s.text}`);

test("the story is the generated roots' texts with the marker between them; video clips and left-out scenes are not in it", () => {
  const p = proj(); p.scenes[3].excluded = true;
  assert.deepEqual(storyRoots(p).map((s) => s.id), ["a", "b"]);
  assert.equal(joinStory(p, "qcut"), "one\nqcut\ntwo");
});

test("match: exact text pins a scene through a reorder; the rest take the leftovers in order", () => {
  assert.deepEqual(match(["three", "one", "new"], ["one", "two", "three"]), [2, 0, 1]);
  assert.deepEqual(match(["x", "two"], ["one", "two"]), [0, 1]);
  assert.deepEqual(match(["a", "b", "c"], ["a"]), [0, -1, -1]);
});

test("editing a scene's words keeps the scene (its id, so its render and anchor)", () => {
  const p = proj();
  assert.equal(applyStory(p, ["one!", "two", "three"]), true);
  assert.deepEqual(texts(p), ["a:one!", "b:two", "v:", "c:three"]);
});

test("a dropped part removes its scene (a rendered one leaves a ghost); the video clip stays", () => {
  const p = proj(); p.scene_renders.b = { media: { filename: "b.mp4" } };
  assert.equal(applyStory(p, ["one", "three"]), true);
  assert.deepEqual(texts(p), ["a:one", "v:", "c:three"]);
  assert.deepEqual(p.scene_ghosts.map((g) => g.id), ["b"]);
});

test("a part nothing owns becomes a new scene at the end of the story", () => {
  const p = proj();
  assert.equal(applyStory(p, ["one", "two", "three", "four"]), true);
  assert.deepEqual(p.scenes.map((s) => s.text), ["one", "two", "", "three", "four"]);
});

test("a part with new words takes over the leftover scene in place, so its render and anchor are kept", () => {
  const p = proj();
  applyStory(p, ["one", "extra", "three"]);
  assert.deepEqual(texts(p), ["a:one", "b:extra", "v:", "c:three"]);
});

test("a reordered story reorders the plan; an unchanged story changes nothing", () => {
  const p = proj();
  assert.equal(applyStory(p, ["three", "one", "two"]), true);
  assert.deepEqual(p.scenes.map((s) => s.id), ["c", "a", "v", "b"]);       // the video clip keeps its slot
  const q = proj();
  assert.equal(applyStory(q, ["one", "two", "three"]), false);
});

test("an empty split changes nothing", () => assert.equal(applyStory(proj(), []), false));

test("an emptied box does not delete the scenes", () => {
  const p = proj(), before = JSON.stringify(p);
  assert.equal(applyStory(p, [""]), false);
  assert.equal(JSON.stringify(p), before);
});

test("editing one scene leaves the plan order alone, even with a montage or a cut unit interleaved", () => {
  const p = { scenes: [{ id: "A", text: "alpha" }, { id: "B", text: "beta" }, { id: "a1", gen_unit_id: "A", cut_offset_frames: 40, text: "" }, { id: "b1", gen_unit_id: "B", cut_offset_frames: 40, text: "" }, { id: "a2", gen_unit_id: "A", cut_offset_frames: 0, text: "alpha" }], scene_renders: {}, scene_ghosts: [] };
  applyStory(p, ["alpha!", "beta"]);
  assert.deepEqual(p.scenes.map((s) => s.id), ["A", "B", "a1", "b1", "a2"]);
  assert.equal(p.scenes.find((s) => s.id === "a2").text, "alpha!", "the first piece shares the prompt");
});

test("dropping a cut scene's paragraph removes the scene with its cuts instead of promoting a cut", () => {
  const p = { scenes: [{ id: "A", text: "alpha" }, { id: "A2", gen_unit_id: "A", cut_offset_frames: 40, text: "" }, { id: "B", text: "beta" }], scene_renders: {}, scene_ghosts: [] };
  assert.equal(applyStory(p, ["beta"]), true);
  assert.deepEqual(p.scenes.map((s) => s.id), ["B"]);
});

test("clash finds a scene whose own text contains a cut word", () => {
  const p = proj(); p.scenes[1].text = "a quick qcut to b";
  assert.equal(clash(p, ["qcut"]).id, "b");
  assert.equal(clash(p, ["other"]), null);
});

const manual = (ids, extra = {}) => ({ scenes: ids.map((id) => ({ id, text: id })), scene_renders: {}, scene_ghosts: [], timeline_manually_ordered: true, timeline_order: [...ids], ...extra });
const shown = (p) => p.timeline_order.map((id) => p.scenes.find((s) => s.id === id)?.text ?? `?${id}`);

test("a manually ordered timeline gets new scenes where the story puts them, not at the end", () => {
  const p = manual(["two"]);
  applyStory(p, ["one", "two", "three", "four"]);
  assert.deepEqual(p.scenes.map((s) => s.text), ["one", "two", "three", "four"]);
  assert.deepEqual(shown(p), ["one", "two", "three", "four"]);
});

test("a manually ordered timeline keeps its own order for old scenes while new ones sit next to their story neighbours", () => {
  const p = manual(["a", "b", "c"], { timeline_order: ["c", "a", "b"] });
  applyStory(p, ["a", "new", "b", "c"]);
  assert.deepEqual(shown(p), ["c", "a", "new", "b"]);
});

test("a first scene typed gradually stays first (cut word added, then the second scene)", () => {
  const p = manual(["one"]);
  applyStory(p, ["one", ""]);
  applyStory(p, ["one", "two"]);
  applyStory(p, ["one", "two", "three"]);
  assert.deepEqual(shown(p), ["one", "two", "three"]);
});

test("scenes are followed by id: edited AND moved scenes keep their render and anchor", () => {
  const p = { scenes: [{ id: "a", text: "a red fox runs through snow" }, { id: "b", text: "a blue boat drifts on a calm lake" }], scene_renders: { a: { media: { filename: "a.mp4" } }, b: { media: { filename: "b.mp4" } } }, scene_ghosts: [] };
  const shown = shownStory(p);
  assert.equal(applyStory(p, ["a blue boat sails on a calm lake at dusk", "a red fox leaps through deep snow"], shown), true);
  assert.deepEqual(p.scenes.map((s) => s.id), ["b", "a"]);
  assert.equal(p.scenes[0].text, "a blue boat sails on a calm lake at dusk");
  assert.deepEqual(p.scene_ghosts, []);
});

test("two scenes with the same words stay two scenes with their own ids; a changed neighbour does not shift them", () => {
  const p = { scenes: [{ id: "a", text: "same" }, { id: "b", text: "same" }, { id: "c", text: "other" }], scene_renders: {}, scene_ghosts: [] };
  assert.equal(applyStory(p, ["same", "same", "other words entirely"], shownStory(p)), true);
  assert.deepEqual(p.scenes.map((s) => `${s.id}:${s.text}`), ["a:same", "b:same", "c:other words entirely"]);
});

test("the shown list, not the plan, decides who a part is when the plan moved meanwhile", () => {
  const p = proj();
  const shown = shownStory(p);
  p.scenes = [p.scenes[1], p.scenes[0], p.scenes[2], p.scenes[3]];       // reordered elsewhere
  applyStory(p, ["one!", "two", "three"], shown);
  assert.deepEqual(Object.fromEntries(p.scenes.map((s) => [s.id, s.text])), { a: "one!", b: "two", v: "", c: "three" });       // the ids keep their words; the order is what the box showed
});
