// Renders live in ComfyUI's temp folder, which ComfyUI empties when it starts. A project opened afterwards still names them:
// this asks the server which files are gone and forgets those renders and takes, so the clips read "not made yet" instead of failing to play.
import { composer as c } from "../../composer/composer.js";
import { keyOf, renderFiles, forgetGone } from "../../shell/renders.js";

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
  setup({ app }) {
    const look = async () => {
      const n = await forgetGoneRenders(app.project, app.api);
      if (n) c.toast.warn({ text: `${n} generated file(s) of this project are gone (ComfyUI removes them when it restarts). Their clips can be generated again.` });
    };
    look();
    return app.on((what) => { if (what === "open") look(); });
  },
};
