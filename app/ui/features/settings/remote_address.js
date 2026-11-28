// What a person pastes or types for the remote machine: "host", "host:port", "user@host:port", or the whole line a rental site
// gives ("ssh -p 31189 root@ssh4.vast.ai -L 8080:localhost:8080"). Returns the fields it could read.

export function parseAddress(text) {
  const out = {};
  let rest = String(text || "").trim();
  if (!rest) return out;
  const words = rest.split(/\s+/);
  if (words[0] === "ssh" || words.some((w) => w === "-p" || w === "-i")) {
    const keep = [];
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (w === "ssh") continue;
      if (w === "-p" && words[i + 1]) { out.port = Number(words[++i]); continue; }
      if (w === "-i" && words[i + 1]) { out.key = words[++i]; continue; }
      if (w === "-L" || w === "-o" || w === "-l") { i++; continue; }                     // the tunnel is made here; the rest is not needed
      if (/^-p\d+$/.test(w)) { out.port = Number(w.slice(2)); continue; }
      if (w.startsWith("-")) continue;
      keep.push(w);
    }
    rest = keep[0] || "";
  }
  const at = rest.lastIndexOf("@");
  if (at > 0) { out.user = rest.slice(0, at); rest = rest.slice(at + 1); }
  const m = rest.match(/^\[?([^\]]+?)\]?(?::(\d{1,5}))?$/);                            // host, host:port, [v6]:port
  if (m && !(rest.split(":").length > 2 && !rest.startsWith("["))) { out.host = m[1]; if (m[2]) out.port = Number(m[2]); }
  else out.host = rest;
  if (out.port != null && !(out.port > 0 && out.port < 65536)) delete out.port;
  return out;
}

export const addressText = ({ host, port } = {}) => (host ? `${host}${port && port !== 22 ? `:${port}` : ""}` : "");
