// The reference sheet dialog's state, apart from the screen: how many slots, what is in each, and the request the server gets.
export const FORMATS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"];
export const DEFAULTS = { title: "", rows: 2, cols: 3, format: "1:1", width: 512 };
export const MAX_SLOTS = 36;

/** The slots for a grid of rows x cols: what was already there stays in its place (reading order), new ones are empty. */
export function resize(slots, rows, cols) {
  return Array.from({ length: rows * cols }, (_, i) => slots[i] || { picture: "", label: "" });
}

/** The server's request. Trailing empty slots are left out; a slot with only a label still keeps its place. */
export function request(state) {
  const slots = state.slots.map((s) => ({ picture: s.picture || undefined, label: s.label.trim() || undefined }));
  while (slots.length && !slots[slots.length - 1].picture && !slots[slots.length - 1].label) slots.pop();
  return { title: state.title.trim(), rows: state.rows, cols: state.cols, format: state.format, width: state.width, slots };
}

/** Why the sheet cannot be made yet, or "". */
export function problem(state) {
  if (state.rows * state.cols > MAX_SLOTS) return `${state.rows} x ${state.cols} is more than ${MAX_SLOTS} slots.`;
  return state.slots.some((s) => s.picture) ? "" : "Pick at least one picture.";
}
