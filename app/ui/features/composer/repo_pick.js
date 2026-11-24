// Which files of a Hugging Face repo are picked for the engine model after one toggle.
// A model file replaces another model file (a second quantization is another model, not a part of the first); the parts of one split model
// go together; a picture projector rides along with whichever model is picked; an importance matrix is no model and never picked.
const AUX = /mmproj/i, NEVER = /imatrix/i;
const partsKey = (n) => String(n).replace(/\d{3,}-of-\d{3,}/i, "#");

/** `all`: the repo's file names, `picked`: the names picked now, `name`/`on`: the toggle. -> the names to keep, the model first. */
export function pickFiles(all, picked, name, on) {
  if (NEVER.test(name)) return picked.filter((n) => all.includes(n));
  let next = all.filter((n) => (n === name ? on : picked.includes(n)) && !NEVER.test(n));
  if (on && !AUX.test(name)) next = next.filter((n) => n === name || AUX.test(n) || partsKey(n) === partsKey(name));
  return [...next.filter((n) => !AUX.test(n)), ...next.filter((n) => AUX.test(n))];
}
