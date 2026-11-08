// Sizes and gaps come from the scale in tokens.css (--sp-*, --ctl-h-*, --lane-*, ...), not from literals in element
// rules: a literal is a number nobody can change in one place. There are none left. Exempt by design: a 1px hairline
// (that is --bw's own value), 0, token definitions (a line starting `--`, where a number belongs), and the width in a
// @media / @container condition (a custom property cannot be used there; the breakpoints are listed in tokens.css).
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const dir = join(dirname(fileURLToPath(import.meta.url)), "..", "elements");
const frame = join(dir, "..", "..", "frame.css");                  // the page frame is held to the same scale
const sheets = () => [...readdirSync(dir).filter((n) => n.endsWith(".css")).map((n) => join(dir, n)), frame];

test("element CSS has no literal pixel sizes", () => {
  let found = 0;
  for (const f of sheets()) {
    const css = readFileSync(f, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
    for (const decl of css.split(/[;{}]/)) {
      if (/^\s*(--|@media|@container)/.test(decl)) continue;       // a token definition is where a number belongs; a breakpoint cannot use one
      found += (decl.match(/(?<![\w.])-?(?:[2-9]|\d{2,})(?:\.\d+)?px\b/g) || []).length;     // 1px lines and 0 are borders and resets
    }
  }
  assert.equal(found, 0, `${found} literal px values in element CSS: use a --sp-* step or a component size from tokens.css`);
});

test("element CSS has no literal colours outside token definitions", () => {
  const bad = [];
  for (const f of sheets()) {
    const css = readFileSync(f, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
    for (const decl of css.split(/[;{}]/)) if (!/^\s*--/.test(decl) && /#[0-9a-fA-F]{3,8}\b/.test(decl)) bad.push(`${f}: ${decl.trim().slice(0, 60)}`);
  }
  assert.deepEqual(bad, [], "literal colours in element CSS: use a token from tokens.css");
});
