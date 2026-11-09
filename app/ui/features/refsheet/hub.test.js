import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { offer, _reset } from "../../shell/mounts.js";
import { loadFeatures } from "../../shell/features.js";
import { problem, request, resize } from "./sheet.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

const slot = (picture = "", label = "") => ({ picture, label });

test("resizing keeps what is in place (reading order) and the request leaves trailing empty slots out", () => {
  const two = [slot("a", "front"), slot("", "back")];
  const six = resize(two, 2, 3);
  assert.equal(six.length, 6);
  assert.deepEqual(six.slice(0, 2), two);
  assert.deepEqual(resize(six, 1, 2), two);
  const r = request({ title: " Mara ", rows: 2, cols: 3, format: "4:3", width: 512, slots: six });
  assert.equal(r.title, "Mara");
  assert.deepEqual(r.slots, [{ picture: "a", label: "front" }, { picture: undefined, label: "back" }]);
});

test("a sheet needs one picture and no more than 36 slots", () => {
  assert.match(problem({ rows: 2, cols: 2, slots: [slot("", "x")] }), /at least one picture/);
  assert.equal(problem({ rows: 2, cols: 2, slots: [slot("a")] }), "");
  assert.match(problem({ rows: 12, cols: 12, slots: [slot("a")] }), /36/);
});

test("it is hidden when the server has no refsheet module, and the button opens a dialog that sends the grid and leaves nothing behind", async () => {
  _reset();
  const host = document.createElement("div");
  offer("assets.actions", host);
  let sent;
  const said = [];
  const api = { media: async () => ({ media: [{ id: "a", name: "front.png", kind: "image" }, { id: "v", name: "clip.mp4", kind: "video" }] }), uploadMedia: async () => ({ media: [], problems: [] }),
    refsheetMake: async (b) => { sent = b; return { media: { id: "n", name: "Sheet.png" }, missing: [] }; } };
  const app = { api, dom: { pickFile: async () => null }, say: (w) => said.push(w), hasModule: () => false };
  const none = await loadFeatures(["x"], { load: async () => ({ default: hub }), app, has: (n) => n in app });
  assert.equal(none.mounted.length, 0);

  _reset();
  offer("assets.actions", host);
  app.hasModule = () => true;
  const { mounted } = await loadFeatures(["x"], { load: async () => ({ default: hub }), app, has: (n) => n in app });
  assert.equal(mounted.length, 1);
  const button = host.querySelector("button");
  assert.match(button.textContent, /Reference sheet/);
  button.click();
  await new Promise((r) => setTimeout(r, 20));
  const buttons = (text) => [...document.querySelectorAll("button")].filter((b) => b.textContent.includes(text));
  assert.equal(buttons("Choose picture…").length, 6);
  assert.ok(document.body.textContent.includes("Rows") && document.body.textContent.includes("Columns") && document.body.textContent.includes("Slot format"));      // every control has a visible label
  buttons("Choose picture…")[0].click();
  await new Promise((r) => setTimeout(r, 20));
  const tiles = [...document.querySelectorAll("[role=option]")];
  assert.deepEqual(tiles.map((t) => t.textContent), ["front.png"]);                 // the bin's pictures only: a video is not offered
  assert.ok(tiles[0].querySelector("img").getAttribute("src").endsWith("/a/thumb"));
  tiles[0].click();
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(buttons("Change: front.png").length, 1);
  const make = [...document.querySelectorAll("button")].find((b) => b.textContent === "Make sheet");
  make.click();
  await new Promise((r) => setTimeout(r, 20));
  assert.deepEqual(sent, { title: "", rows: 2, cols: 3, format: "1:1", width: 512, slots: [{ picture: "a", label: undefined }] });
  assert.ok(said.includes("media"));                  // the media bin redraws on "media"
  mounted[0].teardown();
  assert.equal(host.textContent, "");
});
