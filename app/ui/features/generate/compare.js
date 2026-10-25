// Quick A/B's result: the clip that was on the scene before and the new one, labelled, side by side.
import { composer as c } from "../../composer/composer.js";
import { viewUrl } from "../../shell/run.js";

/** pairs: [{ before, after }] media of each scene made ({filename, subfolder, type}); the labels say how the two differ. */
export function showCompare({ pairs, seed }) {
  const viewers = [];
  const column = (title, media) => { const v = c.viewer.media({ src: viewUrl(media), kind: "video", loop: true }); v.setLook({ fit: "contain" }); viewers.push(v); return c.region.stack({ gap: "sm", children: [c.label.section({ text: title }), v] }); };
  const rows = pairs.map(({ before, after }) => c.splitPane.h({ size: 50, label: "Resize the two clips", panes: [column("Before: as it was", before), column(`After: enhancements off, seed ${seed}`, after)] }));
  const playBoth = () => viewers.forEach((v) => { const m = v.element; if (m) { m.currentTime = 0; m.play().catch(() => {}); } });
  const win = c.modal.generic({ title: "Quick A/B", size: "xl", body: c.region.stack({ gap: "md", children: [...rows, c.hint.default({ text: "Both clips come from the same seed; only the enhancements differ." })] }) });
  win.setFooter({ actions: [c.button.sm({ label: "Play both", tone: "primary", onClick: playBoth }), c.button.sm({ label: "Close", tone: "ghost", onClick: () => win.close("done") })] });
  return win;
}
