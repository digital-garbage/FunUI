import test from "node:test";
import assert from "node:assert/strict";
import { interpolate, modeLabel } from "./job.js";
import { derived, swap, wentThrough } from "../../shell/renders.js";

const m = (f, sub = "") => ({ filename: f, subfolder: sub, type: "output" });

test("the options reach the node, the job is waited for, and an error is thrown in words", async () => {
  const seen = [], seq = [null, { videos: [m("fi.mp4", "funui_interpolated")] }];
  const api = { queueInterpolate: async (...a) => (seen.push(a), "id"), upscaleResult: async () => seq.shift() };
  assert.equal((await interpolate(api, m("x.mp4"), { multiplier: 2 }, async () => {})).filename, "fi.mp4");
  assert.deepEqual(seen[0][1], { multiplier: 2 });
  const bad = { queueInterpolate: async () => "id", upscaleResult: async () => ({ error: "no libnvidia-ngx" }) };
  await assert.rejects(interpolate(bad, m("x.mp4"), {}, async () => {}), /libnvidia-ngx/);
});

test("modes read as words", () => {
  assert.deepEqual(["LOW", "MEDIUM", "HIGH"].map(modeLabel), ["Fast", "Balanced", "Best"]);
});

test("a file that went through a step is never offered it again, whichever order the two steps ran in", () => {
  const raw = m("a.mp4"), up = m("a_up.mp4", "funui_upscaled"), fi = m("a_x2.mp4", "funui_interpolated"), both = m("a_up_x2.mp4", "funui_interpolated");
  assert.ok(!wentThrough(raw, "upscale") && !wentThrough(raw, "interpolate"));
  assert.ok(wentThrough(up, "upscale") && !wentThrough(up, "interpolate"));
  derived(up, both, "interpolate");                                   // upscaled first, then interpolated
  assert.ok(wentThrough(both, "upscale") && wentThrough(both, "interpolate"));    // so the upscale's "always" cannot take it again: no loop
  assert.ok(wentThrough(fi, "interpolate") && !wentThrough(fi, "upscale"));       // interpolated first: still upscalable once
});

test("the interpolated file replaces the render everywhere it played", () => {
  const p = { scene_renders: { a: { media: m("x.mp4") } }, scene_ghosts: [] };
  assert.equal(swap(p, m("x.mp4"), m("fi.mp4", "funui_interpolated")), 1);
});
