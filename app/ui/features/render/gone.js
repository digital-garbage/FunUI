// Renders live in ComfyUI's temp folder, which ComfyUI empties when it starts and which can be emptied by hand at any time. A project still names them:
// this asks the server which files are gone and forgets those renders and takes (their ratings are kept), so the clips read "not made yet" instead of
// playing from the browser's memory or being continued from by the next shot. Asked when a project opens, when files were deleted in the
// Generated & Output Files window, when ComfyUI comes back after the page lost it, and before every run.
import { composer as c } from "../../composer/composer.js";
import { keyOf, renderFiles, forgetGone } from "../../shell/renders.js";
import { LIVE, LOST } from "../../shell/run.js";

/** Ask once for the open project; -> how many files were forgotten (0 when all are there or the server cannot say). */
export async function forgetGoneRenders(p, api) {
  const open = p.project;
  if (!open) return 0;
  const named = renderFiles(open);
  if (!named.length) return 0;
  let answer;
  try { answer = await api.rendersGone(named); } catch { return 0; }       // no answer is not "gone"
  if (!answer || !answer.known || !(answer.gone || []).length) return 0;
  if (!p.project || p.project.id !== open.id) return 0;                     // another project was opened meanwhile
  const gone = new Set(answer.gone.map((i) => named[i]).filter(Boolean).map(keyOf));
  return p.edit((pr) => forgetGone(pr, gone)) ? gone.size : 0;
}

export default {
  id: "render-gone",
  mount: "menubar.menus",
  needs: ["project", "api"],
  modules: ["render"],
  setup({ app, warn = (text) => c.toast.warn({ text }) }) {       // `warn` is injectable so a test can read what is said
    const look = async () => {
      const n = await forgetGoneRenders(app.project, app.api);
      if (n) warn(`${n} generated file(s) of this project are gone (deleted, or removed when ComfyUI restarted). Their clips can be generated again; their ratings are kept.`);
    };
    look();
    (app.beforeRun ||= []).push(look);                    // a run must not continue from, or put a take beside, a file that is no longer there
    // ComfyUI coming back after the page lost it has usually restarted, and a restart empties the temp folder.
    let lost = false;
    const offRun = app.generate && app.generate.subscribe ? app.generate.subscribe((s) => {
      if (s.connection === LOST) lost = true;
      else if (s.connection === LIVE && lost) { lost = false; look(); }
    }) : () => {};
    const offApp = app.on((what) => { if (what === "open" || what === "files.deleted") look(); });
    return () => { offApp(); offRun(); const i = app.beforeRun.indexOf(look); if (i >= 0) app.beforeRun.splice(i, 1); };
  },
};
