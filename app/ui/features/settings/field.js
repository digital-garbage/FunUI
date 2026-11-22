// One control for one module setting ({type, min, max, step, options, ui, hint, unit, default, when}). Engine settings
// and anything else that edits the module-values tree draw their rows through here.
import { composer as c } from "../../composer/composer.js";
import { clampBlocks } from "../../shell/blocks.js";

export { clampBlocks };

/** A `when` clause names sibling settings on the same module that must hold a value for the row to show. */
export const whenSatisfied = (values, id, when) => !when || Object.entries(when).every(([k, v]) => (Array.isArray(v) ? v : [v]).includes((values[id] || {})[k]));

export function settingRow(id, name, spec, values, onChange, blocks = null) {
  const label = spec.label || name;
  const fits = spec.model_blocks && Number.isInteger(blocks) && blocks > 0;
  const hint = (spec.hint ? spec.hint + (spec.unit ? ` (${spec.unit})` : "") : spec.unit || "") + (fits ? ` This model has blocks 0-${blocks - 1}.` : "");
  const now = values[id] && values[id][name];
  const value = now !== undefined ? now : spec.default;
  const set = (v) => onChange(id, name, fits && typeof v === "string" ? clampBlocks(v, blocks) : v);
  let control;
  if (spec.type === "bool") return c.toggle.default({ label, hint, checked: Boolean(value), onChange: set });       // a toggle carries its own label and hint
  if (spec.type === "int" || spec.type === "float") {
    const slider = (spec.ui || "slider") === "slider" && spec.min != null && spec.max != null;
    const step = spec.step ?? (spec.type === "int" ? 1 : 0.01);
    control = slider ? c.slider.readout({ label, value, min: spec.min, max: spec.max, step, precision: spec.type === "int" ? 0 : 2, onCommit: set })
      : c.number.md({ label, value, min: spec.min, max: spec.max, step, precision: spec.type === "int" ? 0 : undefined, onChange: set });
  } else if (spec.type === "enum") control = c.select.md({ label, value, options: (spec.options || []).map((o) => ({ value: o.value, label: o.label || o.value })), onChange: set });
  else if (spec.type === "color") control = c.color.swatch({ label, value: value || undefined, onChange: set });
  else control = (spec.type === "multiline" ? c.textarea.md({ label, value: value || "", rows: 3, onCommit: set }) : c.input.md({ label, value: value || "", onCommit: set }));
  return c.settingsRow.default({ label, hint, control });
}
