// A stand-in for the FunUI server, just enough for the browser tests: serves app/ui under /fun/app/ui/ and answers
// the API with a small fixture project, a pipeline with a sampler, and Training switched on. Anything else answers {}.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "ui");
const TYPES = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".woff2": "font/woff2", ".svg": "image/svg+xml" };
const json = (v) => JSON.stringify(v);

const PROJECT = {
  id: "p1", name: "Demo cut", num_frames_per_scene: 97, frame_rate: 25, scene_renders: {}, timeline_manually_ordered: false,
  scenes: ["A fox runs through snow", "Close-up of the fox", "Wide shot, aurora sky"].map((text, i) => ({ id: `s${i + 1}`, text, frames_mode: "project" })),
};
const PIPELINE = {
  queueable: true,
  slots: [{ id: "k1", node: "KSampler", inputs: { steps: 20, cfg: 6, sampler_name: "euler" },
    roles: ["steps:Steps", "cfg:CFG scale", "sampler_name:Sampler"].map((r) => ({ at: "generation.sampling", input: r.split(":")[0], label: r.split(":")[1] })) }],
};
const NODES = { KSampler: { widgets: [
  { name: "steps", type: "INT", min: 1, max: 100, default: 20 },
  { name: "cfg", type: "FLOAT", min: 0, max: 30, step: 0.1, default: 6 },
  { name: "sampler_name", type: "COMBO", options: ["euler", "dpmpp_2m", "heun"], default: "euler" }] } };
const ANSWERS = {
  "/fun/api/projects": { projects: [{ id: "p1", name: "Demo cut", updated: 1 }] },
  "/fun/api/projects/p1": PROJECT,
  "/fun/api/pipeline": PIPELINE,
  "/fun/api/nodes": { nodes: NODES },
  "/api/m/training_lora/state": { enabled: true },
  "/fun/api/m/training_lora/state": { enabled: true },
};

export function start(port = 0) {
  const server = http.createServer((req, res) => {
    const url = req.url.split("?")[0];
    if (url === "/" || url.startsWith("/fun/app/ui/")) {
      const file = path.join(ROOT, url === "/" ? "index.html" : url.slice("/fun/app/ui/".length));
      if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
      try { const body = fs.readFileSync(file); res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream" }); return res.end(body); }
      catch { res.writeHead(404); return res.end(); }
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(json(ANSWERS[url] || {}));
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve({ server, url: `http://127.0.0.1:${server.address().port}/` })));
}
