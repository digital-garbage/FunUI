// The two chips at the far end of the menubar: whether the project is saved, and whether ComfyUI answers.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";

export default {
  id: "status",
  mount: "menubar.right",
  needs: ["project", "api"],
  setup({ host, app }) {
    const put = inPlace(host);
    let live = true, remote = null;                  // remote: the machine's name while Remote Instance is connected
    const draw = () => put(...(remote ? [c.chip.warn({ label: "Remote Mode Active", dot: true }).node] : []),
      (app.project.unsaved ? c.chip.warn({ label: "unsaved" }) : c.chip.neutral({ label: "saved" })).node,
      (live ? c.chip.good({ label: "ComfyUI live", dot: true }) : c.chip.danger({ label: "ComfyUI offline", dot: true })).node);
    const ping = async () => {
      try { live = Boolean((await app.api.health()).ok); } catch { live = false; }
      try {
        const r = await app.api.remote(), now = r.state === "on" ? r.target : null;
        if (now) live = (await fetch("/system_stats", { cache: "no-store" }).catch(() => ({ ok: false }))).ok;     // while remote: is THAT ComfyUI answering?
        if (remote !== now && (remote || now) && started) {               // it changed under the page (the connection dropped): show this machine's state again
          try { sessionStorage.setItem("funui.remote.note", now ? "" : "The SSH connection ended. FunUI is back on this computer."); } catch { /* private window */ }
          return location.reload();
        }
        remote = now;
      } catch { /* an older server has no remote route */ }
      started = true;
      draw();
    };
    let started = false;
    try { const note = sessionStorage.getItem("funui.remote.note"); if (note) { sessionStorage.removeItem("funui.remote.note"); c.toast.warn({ text: note }); } } catch { /* private window */ }
    draw(); ping();
    const timers = [setInterval(() => { const now = Boolean(app.project.unsaved); if (now !== saved) { saved = now; draw(); } }, 600), setInterval(ping, 5000)];
    let saved = Boolean(app.project.unsaved);
    const off = app.on(() => { saved = Boolean(app.project.unsaved); draw(); });
    return () => { timers.forEach(clearInterval); off(); };
  },
};
