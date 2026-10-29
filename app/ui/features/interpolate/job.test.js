import test from "node:test";
import assert from "node:assert/strict";
import { interpolate, modeLabel } from "./job.js";
import { derived, swap, wentThrough } from "../../shell/renders.js";

const m = (f, sub = "") => ({ filename: f, subfolder: sub, type: "output" });

test("the options reach the job, the job is waited for, and an error is thrown in words", async () => {
  const seen = [], seq = [{ state: "running" }, { state: "done", media: m("fi.mp4", "funui_interpolated") }];
  const api = { interpolate: async (...a) => (seen.push(a), { job_id: "id" }), interpolateJob: async () => seq.shift() };
  assert.equal((await interpolate(api, m("x.mp4"), { multiplier: 2 }, async () => {})).filename, "fi.mp4");
  assert.deepEqual(seen[0][1], { multiplier: 2 });
  const bad = { interpolate: async () => ({ job_id: "id" }), interpolateJob: async () => ({ state: "error", detail: "no libnvidia-ngx" }) };
  await assert.rejects(interpolate(bad, m("x.mp4"), {}, async () => {}), /libnvidia-ngx/);
});

test("a restart is named, a dropped reply is retried", async () => {
  const gone = { interpolate: async () => ({ job_id: "id" }), interpolateJob: async () => { throw Object.assign(new Error("x"), { status: 404 }); } };
  await assert.rejects(interpolate(gone, m("x.mp4"), {}, async () => {}), /restarted/);
  let n = 0;
  const flaky = { interpolate: async () => ({ job_id: "id" }), interpolateJob: async () => { if (++n < 5) throw new Error("net"); return { state: "done", media: m("u.mp4") }; } };
  assert.equal((await interpolate(flaky, m("x.mp4"), {}, async () => {})).filename, "u.mp4");
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
