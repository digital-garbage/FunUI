import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let shortcuts;
test.before(async () => { setupDom(); shortcuts = (await import("./shortcuts.js")).shortcuts; });
test.after(() => teardownDom());

const one = (name, extra = {}) => ({ name, triggers: [name.toLowerCase()], replacements: ["x"], enabled: true, category: "", sub_category: "", ...extra });
const wait = () => new Promise((r) => setTimeout(r, 10));
const mount = async (api) => { const page = shortcuts({ api }, () => {}); document.body.append(page.node); await wait(); return page; };

test("shortcuts are in real alphabetical order: case ignored, and a paused one is placed by its name, not by its pause mark", async () => {
  const lib = { shortcuts: [one("banana"), one("Apple"), one("cherry", { enabled: false }), one("apple2"), one("Zed")] };
  const page = await mount({ shortcuts: async () => lib });
  assert.deepEqual([...page.node.querySelectorAll(".cx-filter-label")].map((n) => n.textContent), ["Apple", "apple2", "banana", "⏸ cherry", "Zed"]);
  page.node.remove();
});

test("what was typed in the search box is still there after a shortcut is saved", async () => {
  const lib = { shortcuts: [one("banana"), one("Apple"), one("bandana")] };
  const page = await mount({ shortcuts: async () => lib, saveShortcut: async () => lib });
  const search = () => page.node.querySelector(".cx-search");
  search().value = "ban";
  search().dispatchEvent(new window.Event("input", { bubbles: true }));
  assert.deepEqual([...page.node.querySelectorAll(".cx-filter-label")].map((n) => n.textContent), ["banana", "bandana"]);
  page.node.querySelector(".cx-filter-row").click();                                     // opens the editor
  [...document.querySelectorAll("button")].find((b) => b.textContent === "Save").click();
  await wait();
  assert.equal(search().value, "ban", "the search box was rebuilt but kept its text");
  assert.equal(page.node.querySelectorAll(".cx-filter-label").length, 2, "and the list is still filtered");
  page.node.remove();
});

// --- find, replace, common phrases -----------------------------------------------------------

const hit = (name, before, after, count = 1) => ({ name, index: 0, count, before, after });
const press = (label, last = false) => { const all = [...document.querySelectorAll("button, [role=menuitem]")].filter((n) => n.textContent.trim() === label); const b = last ? all.at(-1) : all[0]; assert.ok(b, `no "${label}" button`); b.click(); };
const typeInto = (label, value) => { const i = [...document.querySelectorAll("input")].find((n) => n.getAttribute("aria-label") === label); i.value = value; i.dispatchEvent(new window.Event("input", { bubbles: true })); };
const later = (ms) => new Promise((r) => setTimeout(r, ms));
const rows = () => [...document.querySelectorAll(".cx-modal .cx-filter-row")].map((r) => r.textContent);
const closeAll = () => { for (const m of document.querySelectorAll(".cx-modal")) m.remove(); document.querySelectorAll(".cx-menu").forEach((m) => m.remove()); };

test("Find lists the matches with their counts and sends the text and the case choice as typed", async () => {
  const asked = [];
  const lib = { shortcuts: [one("a")] };
  const page = await mount({ shortcuts: async () => lib, findShortcuts: async (f, mc, ww) => { asked.push([f, mc, ww]); return { matches: [hit("a", "red fox red", null, 2), hit("b", "a red hat", null)], skipped: [] }; } });
  page.node.querySelector("button[title^='Export']").click();
  press("Find and replace…");
  typeInto("Find", "a.b (red)");
  await later(260);
  assert.deepEqual(asked.at(-1), ["a.b (red)", false, true]);
  assert.match(document.querySelector(".cx-modal").textContent, /3 matches in 2 shortcuts/);
  assert.deepEqual(rows(), ["a" + "red fox red", "b" + "a red hat"]);
  assert.ok([...document.querySelectorAll("button")].find((b) => /everywhere/.test(b.textContent)).disabled, "Find alone can change nothing");
  closeAll(); page.node.remove();
});

test("Replace shows before and after, asks first, then saves and redraws with the result", async () => {
  const calls = [];
  let lib = { shortcuts: [one("a")] };
  const api = { shortcuts: async () => lib,
    replaceShortcuts: async (f, r, mc, apply, ww) => { calls.push([f, r, mc, apply, ww]); return { ...(apply ? { shortcuts: [one("renamed")] } : {}), matches: [hit("a", "red fox", "blue fox")], changed: 1, skipped: [] }; } };
  const page = await mount(api);
  page.node.querySelector("button[title^='Export']").click();
  press("Find and replace…");
  press("Replace");
  typeInto("Find", "red"); typeInto("Replace with", "blue");
  await later(260);
  assert.deepEqual(calls.at(-1), ["red", "blue", false, false, true], "a preview writes nothing");
  assert.deepEqual(rows(), ["ared fox → blue fox"]);
  press("Replace everywhere");
  await wait();
  assert.equal(calls.filter((c) => c[3]).length, 0, "nothing is written before the person agrees");
  press("Replace", true);                                                                           // the confirmation
  await later(30);
  assert.deepEqual(calls.at(-1), ["red", "blue", false, true, true]);
  assert.deepEqual([...page.node.querySelectorAll(".cx-filter-label")].map((n) => n.textContent), ["renamed"]);
  closeAll(); page.node.remove();
});

test("an empty Find searches nothing, and an empty Replace is offered as a removal", async () => {
  let asked = 0;
  const page = await mount({ shortcuts: async () => ({ shortcuts: [one("a")] }), replaceShortcuts: async () => { asked++; return { matches: [], skipped: [] }; } });
  page.node.querySelector("button[title^='Export']").click();
  press("Find and replace…");
  press("Replace");
  typeInto("Find", "   ");
  await later(260);
  assert.equal(asked, 0);
  assert.ok([...document.querySelectorAll("button")].some((b) => b.textContent === "Remove everywhere"));
  closeAll(); page.node.remove();
});

test("Common phrases lists them with counts, and picking one opens Replace with it filled in", async () => {
  const asked = [];
  const page = await mount({ shortcuts: async () => ({ shortcuts: [one("a")] }),
    shortcutPhrases: async () => ({ phrases: [{ phrase: "golden light", shortcuts: 3, uses: 4, words: 2 }] }),
    replaceShortcuts: async (f) => { asked.push(f); return { matches: [hit("a", "golden light", "x")], skipped: [] }; } });
  page.node.querySelector("button[title^='Export']").click();
  press("Common phrases…");
  await wait();
  assert.deepEqual(rows(), ["golden light3 shortcuts, 4 lines"]);
  document.querySelector(".cx-modal .cx-filter-row").click();
  await later(30);
  assert.equal([...document.querySelectorAll("input")].find((n) => n.getAttribute("aria-label") === "Find").value, "golden light");
  assert.deepEqual(asked, ["golden light"]);
  closeAll(); page.node.remove();
});

test("the count in the confirmation is asked again for what is typed at that moment", async () => {
  let latest = "";
  const api = { shortcuts: async () => ({ shortcuts: [one("a")] }),
    replaceShortcuts: async (f, _r, _m, apply) => { latest = f; return { matches: f === "red" ? [hit("a", "red", "blue"), hit("b", "red", "blue")] : [hit("a", "r", "b")], skipped: [], changed: 1, shortcuts: apply ? [one("a")] : undefined }; } };
  const page = await mount(api);
  page.node.querySelector("button[title^='Export']").click();
  press("Find and replace…");
  press("Replace");
  typeInto("Find", "r"); await later(260);
  typeInto("Find", "red");                      // typed, and Replace pressed before the pause ends
  press("Remove everywhere");
  await later(40);
  assert.equal(latest, "red");
  assert.match(document.body.textContent, /Change 2 lines/);
  closeAll(); page.node.remove();
});

test("Whole words only is on to begin with and is sent as chosen", async () => {
  const sent = [];
  const page = await mount({ shortcuts: async () => ({ shortcuts: [one("a")] }), findShortcuts: async (f, mc, ww) => { sent.push(ww); return { matches: [], skipped: [] }; } });
  page.node.querySelector("button[title^='Export']").click();
  press("Find and replace…");
  typeInto("Find", "cat"); await later(260);
  const box = document.querySelectorAll(".cx-modal input[type=checkbox]")[1];
  box.click(); await later(260);
  assert.deepEqual(sent, [true, false]);
  closeAll(); page.node.remove();
});
