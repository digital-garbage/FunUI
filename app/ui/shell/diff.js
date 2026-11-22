// A line diff for system prompts. A prompt is mostly one long paragraph, so it is cut into sentences first: a changed word then
// shows as one changed sentence, not as the whole paragraph.

/** The pieces a text is compared by: lines, and inside a line its sentences. */
export function pieces(text) {
  return String(text || "").split("\n").flatMap((line) => (line.trim() ? line.split(/(?<=[.!?])\s+/) : [""]));
}

/** -> [{ t: " " | "-" | "+", text }]: what stays, what goes and what is new, in reading order. */
export function lineDiff(before, after) {
  const a = pieces(before), b = pieces(after), n = a.length, m = b.length;
  const len = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) len[i][j] = a[i] === b[j] ? len[i + 1][j + 1] + 1 : Math.max(len[i + 1][j], len[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ t: " ", text: a[i] }); i++; j++; }
    else if (len[i + 1][j] >= len[i][j + 1]) out.push({ t: "-", text: a[i++] });
    else out.push({ t: "+", text: b[j++] });
  }
  while (i < n) out.push({ t: "-", text: a[i++] });
  while (j < m) out.push({ t: "+", text: b[j++] });
  return out;
}
