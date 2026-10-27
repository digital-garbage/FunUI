// Temporal style "auto": reads a shot's words and picks how its motion should feel (ported from v4's director).
// Energy words want punchier steps, stillness words want a hold, loop words want the end to join the start.
const ENERGY = ["run", "running", "sprint", "chase", "fight", "punch", "kick", "explosion", "explode", "burst", "jump", "leap", "dash", "race",
  "fast", "rapid", "quick", "frantic", "spin", "whirl", "fall", "crash", "swing", "throw", "action", "dance", "dancing", "gallop", "charge",
  "shatter", "blast", "speeding"];
const STILL = ["still", "stillness", "motionless", "static", "frozen", "freeze", "portrait", "calm", "serene", "quiet", "slow", "slowly", "gentle",
  "gradual", "drift", "drifting", "float", "floating", "hover", "stare", "gaze", "meditat", "sleep", "resting", "rest", "standing", "posed", "pose",
  "close-up", "closeup", "macro", "landscape", "ambient", "tranquil", "lingering"];
const LOOP = ["loop", "looping", "seamless", "cycle", "cyclic", "repeat", "repeating", "endless", "perpetual", "continuous loop"];

const count = (text, words) => words.filter((w) => text.includes(w)).length;

/** -> one of the concrete Temporal styles for these words. */
export function styleFor(words) {
  const t = String(words || "").toLowerCase();
  if (!t.trim()) return "natural";
  if (count(t, LOOP)) return "loop";
  const net = count(t, STILL) - count(t, ENERGY);
  if (net === 0) return "natural";
  if (net < 0) return "decelerate";
  return Math.min(2.0, 1.0 + 0.35 * net) >= 1.7 ? "freeze" : "accelerate";
}
