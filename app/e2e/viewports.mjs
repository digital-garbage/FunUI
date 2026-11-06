// The app at the screens people really have, in a real browser: no page overflow, every window inside the viewport,
// both timeline lanes visible. Not part of `npm test` (it needs browsers): run it with `npm run test:viewport`.
// FUNUI_BROWSERS=chromium,firefox picks the engines (default: every one that is installed).
import playwright from "playwright";
import { start } from "./stub.mjs";

const SIZES = [[1280, 720], [1024, 1024], [1366, 768], [1920, 1080], [2560, 1440], [768, 1024], [720, 1280], [1080, 1920]];
const WINDOWS = [["Settings", /^Settings$/], ["Log", /^Log$/], ["Composer", /Composer/], ["Sampler", /Sampler/], ["Training", /Training/]];
const wanted = (process.env.FUNUI_BROWSERS || "chromium,firefox").split(",");

const inspect = () => {
  const box = (e) => e.getBoundingClientRect();
  const seen = [...document.querySelectorAll(".cx-modal, .cx-popover, .cx-window")].filter((e) => box(e).width > 0);
  const out = seen.map((e) => ({ name: e.className.toString().split(" ")[0], r: box(e) }));
  const nle = document.querySelector(".cx-nle"), lanes = [...document.querySelectorAll(".cx-nle-gutter-lane")];
  return {
    over: [document.documentElement.scrollWidth - innerWidth, document.documentElement.scrollHeight - innerHeight],
    windows: out.filter(({ r }) => r.left < -1 || r.top < -1 || r.right > innerWidth + 1 || r.bottom > innerHeight + 1).map(({ name, r }) => `${name} ${Math.round(r.left)},${Math.round(r.top)}..${Math.round(r.right)},${Math.round(r.bottom)}`),
    lanes: lanes.length < 2 ? ["fewer than two lanes"] : lanes.filter((l) => box(l).bottom > box(nle).bottom + 1).map((l) => `lane ends ${Math.round(box(l).bottom)}, timeline ${Math.round(box(nle).bottom)}`),
  };
};

const { server, url } = await start(0);
const problems = [];
let ran = 0;
for (const name of wanted) {
  let browser;
  try { browser = await playwright[name].launch(); } catch { console.log(`skipping ${name}: not installed`); continue; }
  for (const [w, h] of SIZES) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    const note = (what) => problems.push(`${name} ${w}x${h}: ${what}`);
    await page.goto(url);
    await page.getByText(/Continue with/).click();
    await page.waitForTimeout(800);
    const main = await page.evaluate(inspect);
    if (main.over[0] > 0 || main.over[1] > 0) note(`page overflows by ${main.over}`);
    main.lanes.forEach((m) => note(m));
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
