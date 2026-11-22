// Block lists ("2-10,29-46") as modules and the helper write them.
/** A block list ("2-10,29-46") held to a model's blocks 0..count-1: entries past the end are cut, a range is shortened, an entry wholly past the end is dropped. */
export function clampBlocks(text, count) {
  if (!Number.isInteger(count) || count < 1) return text;
  const last = count - 1, out = [];
  for (const part of String(text || "").split(",").map((x) => x.trim()).filter(Boolean)) {
    const m = /^(\d+)(?:\s*-\s*(\d+))?$/.exec(part);
    if (!m) { out.push(part); continue; }             // not a number or range: left for the module to say
    const lo = Number(m[1]), hi = m[2] === undefined ? lo : Number(m[2]);
    const a = Math.min(lo, hi), b = Math.min(Math.max(lo, hi), last);
    if (a > last) continue;
    out.push(a === b ? String(a) : `${a}-${b}`);
  }
  return out.join(",");
}
