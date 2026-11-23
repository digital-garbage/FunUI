// The app at the screens people really have, in a real browser: no page overflow, every window inside the viewport,
// both timeline lanes visible, the timeline header on one row from 1024px wide, idle and while a render is running (a fake ComfyUI queue and socket). Not part of `npm test` (it needs browsers): run it with `npm run test:viewport`.
// FUNUI_BROWSERS=chromium,firefox picks the engines (default: every one that is installed).
import playwright from "playwright";
import { start } from "./stub.mjs";

const SIZES = [[1280, 720], [1024, 1024], [1000, 700], [1366, 768], [1920, 1080], [2560, 1440], [768, 1024], [720, 1280], [1080, 1920]];
const WINDOWS = [["Settings", /^Settings$/], ["Log", /^Log$/], ["Composer", /Composer/], ["Sampler", /Sampler/], ["Training", /Training/]];
const CLIENT = "viewport-test";
const wanted = (process.env.FUNUI_BROWSERS || "chromium,firefox").split(",");

const inspect = () => {
  const box = (e) => e.getBoundingClientRect();
  const seen = [...document.querySelectorAll(".cx-modal, .cx-popover, .cx-window")].filter((e) => box(e).width > 0);
  const out = seen.map((e) => ({ name: e.className.toString().split(" ")[0], r: box(e) }));
  const nle = document.querySelector(".cx-nle"), lanes = [...document.querySelectorAll(".cx-nle-gutter-lane")];
  const head = [...document.querySelectorAll(".cx-panel-head")].find((h) => h.textContent.startsWith("Timeline")), act = head && head.querySelector(".cx-panel-actions");
  const header = innerWidth < 1024 || !head ? [] : [box(head).height > 50 && `Timeline header is ${Math.round(box(head).height)}px tall (more than one row)`, act.scrollWidth > act.clientWidth + 1 && `Timeline header clips ${act.scrollWidth - act.clientWidth}px of its controls`].filter(Boolean);
  return {
    header,
    over: [document.documentElement.scrollWidth - innerWidth, document.documentElement.scrollHeight - innerHeight],
    windows: out.filter(({ r }) => r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1).map(({ name, r }) => `${name} ${Math.round(r.left)},${Math.round(r.top)}..${Math.round(r.right)},${Math.round(r.bottom)}`),
    lanes: lanes.length < 2 ? ["fewer than two lanes"] : lanes.filter((l) => box(l).bottom > box(nle).bottom + 1).map((l) => `lane ends ${Math.round(box(l).bottom)}, timeline ${Math.round(box(nle).bottom)}`),
    slack: nle && nle.querySelector(".cx-nle-stage") && nle.clientHeight - nle.querySelector(".cx-nle-stage").offsetHeight > 6 ? [`${Math.round(nle.clientHeight - nle.querySelector(".cx-nle-stage").offsetHeight)}px of empty space under the last lane`] : [],
  };
};

// A render in progress, as ComfyUI would show it: the page's own client id in the queue, and the socket reporting step 4 of 5.
async function fakeRunning(page) {
  await page.addInitScript((id) => sessionStorage.setItem("funui.client-id", id), CLIENT);
  await page.route("**/queue", (route) => route.fulfill({ json: { queue_pending: [], queue_running: [[1, "pid1", {}, { client_id: CLIENT, funui_scene_id: "s1", funui_scene_ids: ["s1"], funui_project_id: "p1" }, []]] } }));
  await page.routeWebSocket(/\/ws\?/, (ws) => {
    const say = (type, data) => ws.send(JSON.stringify({ type, data }));
    setTimeout(() => { say("execution_start", { prompt_id: "pid1" }); say("progress_state", { prompt_id: "pid1", nodes: { 3: { state: "running", value: 4, max: 5, node_id: "3", display_node_id: "3" } } }); }, 300);
  });
}

// Help > Start tour, then Next to the end: every callout must find its target (a step whose target is missing is skipped, so the numbers would jump).
async function walkTour(page, note) {
  await page.getByRole("button", { name: "Help", exact: true }).click();
  await page.getByText("Start tour").click();
  const seen = [];
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(150);
    const head = await page.evaluate(() => (document.querySelector(".cx-popover .cx-label, .cx-popover") || {}).textContent || "");
    const m = head.match(/(\d+) \/ (\d+)/);
    if (!m) break;
    seen.push(Number(m[1]));
    if (/· Generate/.test(head)) {         // the pointed-at button is redrawn (as when a run starts): the callout must follow the new one
      await page.evaluate(() => { const b = [...document.querySelectorAll("button")].find((x) => (x.textContent || "").trim().startsWith("▶ Generate")); if (b) { const n = b.cloneNode(true); n.textContent = "Generating…"; b.replaceWith(n); } window.dispatchEvent(new Event("resize")); });
      await page.waitForTimeout(400);
      const gap = await page.evaluate(() => {
        const b = [...document.querySelectorAll("button")].find((x) => /^(▶ Generate|Generating…)/.test((x.textContent || "").trim())), pop = document.querySelector(".cx-popover");
        if (!b || !pop || !/· Generate/.test(pop.textContent)) return null;       // still the Generate callout, not skipped on to the next part
        const r = b.getBoundingClientRect(), q = pop.getBoundingClientRect();
        return Math.abs(q.bottom - r.top);       // above the button (the sideways place is clamped to the screen edge)
      });
      if (gap === null || gap > 40) note(`the tour callout did not follow the redrawn Generate button (off by ${gap})`);
    }
    const done = Number(m[1]) === Number(m[2]);
    await page.getByRole("button", { name: done ? "Done" : "Next", exact: true }).click();
    if (done) break;
  }
  if (!seen.length || seen.some((n, i) => n !== i + 1) || seen.length < 5) note(`the tour skipped a step or did not run (saw ${seen.join(",") || "nothing"})`);
}

// Auto-hide: with the pointer resting on the strip the timeline opens, and it must hold its lanes with no empty space under them or past them.
async function autoHide(page, note) {
  await page.getByRole("button", { name: "Auto-hide", exact: true }).click();
  await page.mouse.move(2, 2);
  await page.waitForTimeout(300);
  const strip = await page.locator(".fp-peek-strip").boundingBox();
  if (!strip) return note("auto-hide shows no strip");
  await page.mouse.move(strip.x + strip.width / 2, strip.y + strip.height / 2);
  await page.waitForTimeout(700);
  const gap = await page.evaluate(() => {
    const z = document.querySelector(".fp-main > :nth-child(2)"), nle = z && z.querySelector(".cx-nle"), stage = nle && nle.querySelector(".cx-nle-stage");
    if (!document.documentElement.classList.contains("fp-peek-open") || !nle || !stage) return null;
    const q = (e) => e.getBoundingClientRect();
    return { below: Math.round(q(z).bottom - q(stage).bottom), inside: Math.round(nle.clientHeight - stage.offsetHeight), top: Math.round(q(z).top) };
  });
  if (!gap) note("auto-hide did not open the timeline when the pointer rested on its strip");
  else {
    if (gap.below > 24) note(`auto-hide: ${gap.below}px of empty space under the last lane when hovered`);
    if (gap.inside > 6) note(`auto-hide: ${gap.inside}px of empty space inside the timeline when hovered`);
    if (gap.top < 0) note("auto-hide: the opened timeline runs off the top of the window");
  }
  await page.mouse.move(2, 2);
  await page.getByRole("button", { name: "Auto-hide", exact: true }).click();
  await page.waitForTimeout(200);
}

// The Helper's pinned checklist: a card inside the window with room around its text, that folds and opens, and stays inside the window.
async function helperChecklist(page, note, shot) {
  const items = [["set a", "done"], ["set b", "done"], ["set c", "open"], ["a longer step that needs a good deal more words than the others so the line has to wrap inside the card", "open"], ["set e", "skipped"], ["set f", "open"]]
    .map(([text, state]) => ({ text, state, note: state === "skipped" ? "does not exist" : "" }));
  const events = [{ kind: "begin", message: "rename everything", seconds: 4 }, { kind: "plan", items }, { kind: "thinking", text: "working it out" }, { kind: "answer", text: "I apologize for the earlier mistake, the names I used did not match; I will look them up once more and then rename every one of them with the exact spelling the library shows, in small batches." }, { kind: "doing", tool: "find_shortcuts" }, { kind: "answer", text: "Short note." }, { kind: "doing", tool: "edit_shortcuts" }, { kind: "result", result: { ok: true, reply: "Done." } }];
  await page.route("**/chat_attach**", (route) => route.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("") }));
  await page.getByRole("button", { name: /Helper/ }).first().click({ timeout: 2000 }).catch(() => note("the Helper button cannot be pressed"));
  await page.waitForTimeout(700);
  const got = await page.evaluate(() => {
    const modal = document.querySelector(".cx-modal"), fold = modal && [...modal.querySelectorAll("details.cx-collapsible")].find((d) => d.textContent.startsWith("Checklist")), card = fold;
    if (!modal || !card || !fold) return null;
    const m = modal.getBoundingClientRect(), q = card.getBoundingClientRect(), cs = getComputedStyle(card.querySelector("summary")), boxed = !!fold.closest(".cx-fieldcard, .cx-card"), text = fold.querySelector(".cx-collapsible-body, :scope > :not(summary)");
    const pad = parseFloat(cs.paddingLeft), tx = text ? text.getBoundingClientRect() : null;
    const lines = [...modal.querySelectorAll(".cx-hint")].filter((h) => /apologize|Short note/.test(h.textContent)).map((h) => { const r = h.getBoundingClientRect(); return { inside: r.left >= m.left - 1 && r.right <= m.right + 1, tall: r.height > 120, cut: h.scrollWidth > h.clientWidth + 1 }; });
    const you = [...modal.querySelectorAll(".cx-bubble-you")].find((b) => b.textContent.includes("rename everything")), under = !!you && !!(you.compareDocumentPosition(fold) & Node.DOCUMENT_POSITION_FOLLOWING) && fold.getBoundingClientRect().top >= you.getBoundingClientRect().bottom - 1;
    return { lines, under, boxed, inside: q.left >= m.left - 1 && q.right <= m.right + 1 && q.top >= m.top - 1, pad, open: fold.open, gapLeft: tx ? tx.left - q.left : 0, wide: modal.scrollWidth > modal.clientWidth + 1 };
  });
  if (!got) { note("the Helper's checklist card is not shown"); return; }
  if (got.lines.length !== 2) note(`the Helper's comment lines are not shown (${got.lines.length} of 2)`);
  if (got.lines.some((l) => !l.inside || l.cut)) note("a Helper comment line sticks out of the window");
  if (got.lines.some((l) => l.tall)) note("a Helper comment line is far too tall");
  if (!got.under) note("the checklist is not under its message");
  if (got.boxed) note("the checklist sits in a second box instead of being a plain fold");
  if (!got.inside) note("the checklist card sticks out of the Helper window");
  if (got.pad < 8) note(`the checklist card has no room around its text (${got.pad}px)`);
  if (got.open) note("the checklist of a finished run is not folded");
  if (got.wide) note("the Helper window scrolls sideways");
  const sum = page.locator(".cx-modal details.cx-collapsible", { hasText: /^Checklist/ }).locator("summary");
  const isOpen = () => page.evaluate(() => [...document.querySelectorAll(".cx-modal details.cx-collapsible")].find((d) => d.textContent.startsWith("Checklist")).open);
  await sum.click();
  await page.waitForTimeout(150);
  if (!(await isOpen())) note("the checklist does not open when clicked");
  const high = await page.evaluate(() => { const d = [...document.querySelectorAll(".cx-modal details.cx-collapsible")].find((x) => x.textContent.startsWith("Checklist")).getBoundingClientRect(), m = document.querySelector(".cx-modal").getBoundingClientRect(); return d.right <= m.right + 1 && d.bottom <= innerHeight; });
  if (!high) note("the opened checklist sticks out of the window");
  if (shot) await page.screenshot({ path: shot });
  await sum.click();
  await page.waitForTimeout(150);
  if (await isOpen()) note("the checklist does not fold when clicked");
  await page.keyboard.press("Escape");
  await page.waitForTimeout(200);
}

const { server, url } = await start(0);
const problems = [];
let ran = 0;
for (const name of wanted) {
  let browser;
  try { browser = await playwright[name].launch(); } catch { console.log(`skipping ${name}: not installed`); continue; }
  try {
  for (const [w, h, running] of SIZES.flatMap(([w, h]) => [[w, h, false], [w, h, true]])) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    const note = (what) => problems.push(`${name} ${w}x${h}${running ? " running" : ""}: ${what}`);
    if (running) await fakeRunning(page);
    await page.goto(url);
    await page.getByText(/Continue with/).click();
    await page.waitForTimeout(800);
    if (running) {
      await page.waitForTimeout(700);
      if (!(await page.getByText(/Step 4\/5/).count())) note("the running render's progress is not shown");
      if (process.env.FUNUI_SHOTS) await page.screenshot({ path: `${process.env.FUNUI_SHOTS}/running-${name}-${w}x${h}.png` });
    }
    if (!running) await walkTour(page, note);
    const main = await page.evaluate(inspect);
    if (main.over[0] > 0 || main.over[1] > 0) note(`page overflows by ${main.over}`);
    main.lanes.forEach((m) => note(m));
    main.slack.forEach((m) => note(m));
    main.header.forEach((m) => note(m));
    if (!running) await autoHide(page, note);
    if (!running) await helperChecklist(page, note, process.env.FUNUI_SHOTS ? `${process.env.FUNUI_SHOTS}/helper-${name}-${w}x${h}.png` : "");
    for (const [label, text] of WINDOWS) {
      await page.getByText(text).first().click({ timeout: 1500 }).catch(() => note(`${label} cannot be opened`));
      await page.waitForTimeout(500);
      (await page.evaluate(inspect)).windows.forEach((m) => note(`${label} window outside the viewport: ${m}`));
      await page.keyboard.press("Escape");
      await page.waitForTimeout(200);
    }
    await page.close();
    ran += 1;
  }
  } finally { await browser.close(); }     // a failed run must not leave a browser holding memory on a small machine
}
server.close();
console.log(problems.length ? problems.join("\n") : `ok: ${ran} browser/size runs, no findings`);
process.exit(problems.length || !ran ? 1 : 0);
