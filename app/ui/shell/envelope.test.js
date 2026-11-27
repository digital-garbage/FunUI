import test from "node:test";
import assert from "node:assert/strict";
import { Cipher, utf8, fromUtf8, concat } from "./envelope_cipher.js";
import { createEnvelope } from "./envelope.js";

const cipher = new Cipher(), Q = utf8("q"), R = utf8("r"), W = utf8("w"), origin = "http://box.example:8188";

// A stand-in for the backend's /fun/x: unwraps the request, lets `route` answer it, wraps the answer (the format of core/envelope_http.py).
function backend(route, seen = []) {
  return async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, origin);
    if (url.pathname !== "/fun/x") { seen.push({ plain: url.pathname }); return new Response("plain " + url.pathname); }
    const parts = cipher.unpack(new Uint8Array(await new Response(init.body).arrayBuffer()), Q), meta = JSON.parse(fromUtf8(parts[0])), body = concat(...parts.slice(1));
    seen.push({ meta, body: fromUtf8(body), onWire: fromUtf8(init.body) });
    const [status, headers, out] = await route(meta, body);
    return new Response(concat(cipher.frame(utf8(JSON.stringify({ s: status, h: headers })), R), ...out.map((p) => cipher.frame(p, R))), { status: 200, headers: { "Content-Type": "application/octet-stream" } });
  };
}

test("a call goes to one address, unreadable, and the answer is the real one", async () => {
  const seen = [], env = createEnvelope({ origin, plainFetch: backend(() => [200, { "Content-Type": "application/json" }, [utf8('{"ok":true}')]], seen) });
  const res = await env.fetch("/fun/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: '{"prompt":"a secret dog"}' });
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { ok: true });
  assert.equal(res.headers.get("content-type"), "application/json");
  assert.deepEqual([seen[0].meta.m, seen[0].meta.p, seen[0].body], ["POST", "/fun/api/generate", '{"prompt":"a secret dog"}']);
  assert.ok(!seen[0].onWire.includes("secret") && !seen[0].onWire.includes("generate"));
});

test("an answer written bit by bit arrives bit by bit, and a stop reaches the server", async () => {
  const sent = [utf8("data: 1\n\n"), utf8("data: 2\n\n"), utf8("data: 3\n\n")];
  const wire = concat(cipher.frame(utf8(JSON.stringify({ s: 200, h: { "Content-Type": "text/event-stream" } })), R), ...sent.map((p) => cipher.frame(p, R)));
  // delivered in pieces that cut the frames in odd places
  const pieces = []; for (let i = 0; i < wire.length; i += 7) pieces.push(wire.subarray(i, i + 7));
  const plainFetch = async () => new Response(new ReadableStream({ start(c) { pieces.forEach((p) => c.enqueue(p)); c.close(); } }), { status: 200, headers: { "Content-Type": "application/octet-stream" } });
  const env = createEnvelope({ origin, plainFetch });
  const res = await env.fetch("/fun/api/m/x/chat", { method: "POST", body: "{}" });
  const got = []; const reader = res.body.getReader();
  for (let r = await reader.read(); !r.done; r = await reader.read()) got.push(fromUtf8(r.value));
  assert.deepEqual(got, ["data: 1\n\n", "data: 2\n\n", "data: 3\n\n"]);
});

test("pictures, other origins and the page's own files stay as they were; a server without the address gets plain calls", async () => {
  const seen = [], env = createEnvelope({ origin, plainFetch: backend(() => [200, {}, [utf8("x")]], seen) });
  await env.fetch("/fun/app/ui/boot.js");
  await env.fetch("https://elsewhere.example/api");
  assert.deepEqual(seen.map((s) => s.plain), ["/fun/app/ui/boot.js", "/api"]);
  const old = createEnvelope({ origin, plainFetch: async (input) => (String(input).endsWith("/fun/x") ? new Response("no", { status: 404 }) : new Response("plain answer")) });
  assert.equal(await (await old.fetch("/fun/api/health")).text(), "plain answer");
  assert.ok(old.off());
});

test("a form upload goes wrapped with its progress, and the answer is read", async () => {
  const form = new FormData(); form.append("image", new Blob([new Uint8Array(5000)], { type: "image/png" }), "a.png");
  let wire = null;
  const xhr = () => ({ upload: {}, open() {}, send(w) { wire = w; const part = new Uint8Array(concat(cipher.frame(utf8(JSON.stringify({ s: 200, h: {} })), R), cipher.frame(utf8('{"name":"a.png"}'), R))); this.response = part.buffer; this.upload.onprogress({ lengthComputable: true, loaded: 1, total: 2 }); this.onload(); } });
  const env = createEnvelope({ origin, plainFetch: async () => { throw new Error("no fetch for an upload"); }, plainXhr: xhr });
  const fractions = [], res = await env.upload("/upload/image", form, (f) => fractions.push(f));
  assert.deepEqual([res.ok, res.status, await res.json(), fractions], [true, 200, { name: "a.png" }, [0.5]]);
  const meta = JSON.parse(fromUtf8(cipher.unpack(wire, Q)[0]));
  assert.deepEqual([meta.m, meta.p, /multipart\/form-data; boundary=/.test(meta.h["content-type"])], ["POST", "/upload/image", true]);
});

test("the socket's wrapped frames are heard as the page always heard them, plain ones too", async () => {
  const listeners = {}, real = { binaryType: "", addEventListener: (t, f) => { listeners[t] = f; }, removeEventListener() {}, readyState: 1, close() {} };
  const env = createEnvelope({ origin, plainFetch: async () => new Response("") });
  const heard = [], socket = env.socket(real);
  assert.equal(real.binaryType, "arraybuffer");
  socket.addEventListener("message", (e) => heard.push(e.data));
  socket.addEventListener("open", () => heard.push("open"));
  const send = (kind, bytes) => listeners.message({ data: cipher.seal(concat(Uint8Array.of(kind), bytes), W).buffer });
  send(0, utf8('{"type":"progress"}')); send(1, Uint8Array.of(1, 2, 3));
  listeners.message({ data: '{"type":"plain"}' });
  listeners.open();
  assert.equal(heard[0], '{"type":"progress"}');
  assert.deepEqual([...new Uint8Array(heard[1])], [1, 2, 3]);
  assert.deepEqual(heard.slice(2), ['{"type":"plain"}', "open"]);
  assert.equal(socket.readyState, 1);
});

test("a clip or picture is fetched wrapped and handed over as a blob; a download keeps the server's name", async () => {
  globalThis.URL.createObjectURL ||= () => "blob:fake";
  const env = createEnvelope({ origin, plainFetch: backend(() => [200, { "Content-Type": "video/mp4", "Content-Disposition": "attachment; filename*=UTF-8''clip%20one.mp4" }, [utf8("VIDEO")]]) });
  assert.ok((await env.blobFor("/view?filename=a.mp4")).startsWith("blob:"));
  assert.equal(await env.blobFor("blob:abc"), null);
  const got = await env.download("/fun/api/media/1/file", "");
  assert.equal(got.name, "clip one.mp4");
  assert.equal(await got.blob.text(), "VIDEO");
});
