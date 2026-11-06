// The app at the screens people really have, in a real browser: no page overflow, every window inside the viewport,
// both timeline lanes visible, the timeline header on one row from 1000px wide, idle and while a render is running (a fake ComfyUI queue and socket). Not part of `npm test` (it needs browsers): run it with `npm run test:viewport`.
// FUNUI_BROWSERS=chromium,firefox picks the engines (default: every one that is installed).
import playwright from "playwright";
import { start } from "./stub.mjs";

const SIZES = [[1280, 720], [1024, 1024], [1366, 768], [1920, 1080], [2560, 1440], [768, 1024], [720, 1280], [1080, 1920]];
const WINDOWS = [["Settings", /^Settings$/], ["Log", /^Log$/], ["Composer", /Composer/], ["Sampler", /Sampler/], ["Training", /Training/]];
const CLIENT = "viewport-test";
const wanted = (process.env.FUNUI_BROWSERS || "chromium,firefox").split(",");

const inspect = () => {
  const box = (e) => e.getBoundingClientRect();
  const seen = [...document.querySelectorAll(".cx-modal, .cx-popover, .cx-window")].filter((e) => box(e).width > 0);
  const out = seen.map((e) => ({ name: e.className.toString().split(" ")[0], r: box(e) }));
  const nle = document.querySelector(".cx-nle"), lanes = [...document.querySelectorAll(".cx-nle-gutter-lane")];
  const head = [...document.querySelectorAll(".cx-panel-head")].find((h) => h.textContent.startsWith("Timeline")), act = head && head.querySelector(".cx-panel-actions");
  const header = innerWidth < 1000 || !head ? [] : [box(head).height > 50 && `Timeline header is ${Math.round(box(head).height)}px tall (more than one row)`, act.scrollWidth > act.clientWidth + 1 && `Timeline header clips ${act.scrollWidth - act.clientWidth}px of its controls`].filter(Boolean);
  return {
    header,
    over: [document.documentElement.scrollWidth - innerWidth, document.documentElement.scrollHeight - innerHeight],
    windows: out.filter(({ r }) => r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1).map(({ name, r }) => `${name} ${Math.round(r.left)},${Math.round(r.top)}..${Math.round(r.right)},${Math.round(r.bottom)}`),
    lanes: lanes.length < 2 ? ["fewer than two lanes"] : lanes.filter((l) => box(l).bottom > box(nle).bottom + 1).map((l) => `lane ends ${Math.round(box(l).bottom)}, timeline ${Math.round(box(nle).bottom)}`),
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

const { server, url } = await start(0);
const problems = [];
let ran = 0;
for (const name of wanted) {
  let browser;
  try { browser = await playwright[name].launch(); } catch { console.log(`skipping ${name}: not installed`); continue; }
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
    const main = await page.evaluate(inspect);
    if (main.over[0] > 0 || main.over[1] > 0) note(`page overflows by ${main.over}`);
    main.lanes.forEach((m) => note(m));
    main.header.forEach((m) => note(m));
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
  await browser.close();
}
server.close();
console.log(problems.length ? problems.join("\n") : `ok: ${ran} browser/size runs, no findings`);
process.exit(problems.length || !ran ? 1 : 0);
