// A collapsible group for the Properties panel. The panel redraws on most edits, so whether a group is open is
// remembered here and a redraw does not close what the person just opened.
import { composer as c } from "../../composer/composer.js";

const opened = new Map();

/** `id`: a stable name for the group. `hint`: one short line shown beside the label while it is closed. */
export const fold = (id, label, children, hint) => c.collapsible.default({ label, hint, open: opened.get(id) === true,
  onToggle: (open) => opened.set(id, open), body: c.region.stack({ gap: "sm", children: children.flat().filter(Boolean) }) });
