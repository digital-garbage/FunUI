// Sizes and gaps come from the scale in tokens.css (--sp-*, --ctl-h-*, --lane-*, ...), not from literals in element
// rules: a literal is a number nobody can change in one place. This is a ratchet: the count of literal pixel values
// in element CSS may go down, never up. Lowering it is the point; when it does, lower LITERAL_PX here too.
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const dir = join(dirname(fileURLToPath(import.meta.url)), "..", "elements");
const LITERAL_PX = 118;

test("element CSS does not grow new literal pixel sizes", () => {
  let found = 0;
  for (const f of readdirSync(dir).filter((n) => n.endsWith(".css"))) {
    const css = readFileSync(join(dir, f), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
    for (const decl of css.split(/[;{}]/)) {
      if (/^\s*--/.test(decl)) continue;                          // a token definition is where a number belongs
      found += (decl.match(/(?<![\w.-])(?:[2-9]|\d{2,})(?:\.\d+)?px\b/g) || []).length;     // 1px lines and 0 are borders and resets
    }
  }
  assert.ok(found <= LITERAL_PX, `${found} literal px values in element CSS (limit ${LITERAL_PX}): use a token from tokens.css`);
  if (found < LITERAL_PX) console.log(`lower LITERAL_PX to ${found}`);
});
