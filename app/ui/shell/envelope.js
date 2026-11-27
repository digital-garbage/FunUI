// Light privacy for the traffic between this page and the backend (core/envelope_http.py is the other half).
//
// Every call the page makes to its own server goes to ONE address, /fun/x, as an encrypted message, so the wire shows an
// unreadable body and no feature names; the answer comes back the same way (progress lines are decrypted as they arrive,
// finished media is decrypted into a blob the element plays). The socket's frames are wrapped too. Always on: there is
// nothing to set up. If the server has no such address (an older one) the page quietly uses plain requests.
//
// This is LIGHT PRIVACY, not security: the key is built in and the same everywhere (see envelope_cipher.js).

import { Cipher, utf8, fromUtf8, concat, CHUNK, NONCE, TAG, EnvelopeError } from "./envelope_cipher.js";

const ENDPOINT = "/fun/x";
const Q = utf8("q"), R = utf8("r"), W = utf8("w");
const STATIC = ["/fun/app/", ENDPOINT];                 // the page's own files are plain: nothing in them is said by the person
const NO_BODY = new Set([101, 204, 205, 304]);
const EMPTY = new Uint8Array(0);

/** The decrypted messages of an answer, as they arrive. */
async function* messages(reader, cipher) {
  let buf = EMPTY;
  for (;;) {
    while (buf.length >= 4) {
      const size = new DataView(buf.buffer, buf.byteOffset, buf.length).getUint32(0);
      if (size > CHUNK + NONCE + TAG) throw new EnvelopeError("a message is too long");
      if (buf.length < 4 + size) break;
      yield cipher.unseal(buf.subarray(4, 4 + size), R);
      buf = buf.subarray(4 + size);
    }
    const { value, done } = await reader.read();
    if (done) { if (buf.length) throw new EnvelopeError("a message is cut short"); return; }
    buf = buf.length ? concat(buf, value) : value;
  }
}

/** The answer of /fun/x -> a Response like the one the real address would have given. */
async function unwrap(outer, cipher) {
  const reader = outer.body ? outer.body.getReader() : new Response(await outer.arrayBuffer()).body.getReader();
  const it = messages(reader, cipher), head = await it.next();
  if (head.done) throw new EnvelopeError("the answer was empty");
  const meta = JSON.parse(fromUtf8(head.value));
  const body = NO_BODY.has(meta.s) ? null : new ReadableStream({
    async pull(controller) {
      try { const { value, done } = await it.next(); if (done) controller.close(); else if (value.length) controller.enqueue(value); } catch (err) { controller.error(err); }
    },
    cancel() { return reader.cancel(); },
  });
  if (!body) reader.cancel().catch(() => {});
  return new Response(body, { status: meta.s, headers: meta.h || {} });
}

export function createEnvelope({ cipher = new Cipher(), plainFetch, origin = globalThis.location && globalThis.location.origin, plainXhr } = {}) {
  let off = false;                                       // the server does not know the address: plain requests from here on
  const wrapped = (url) => !off && url.origin === origin && !STATIC.some((p) => url.pathname.startsWith(p));

  async function pack(request) {
    const url = new URL(request.url, origin), headers = {};
    request.headers.forEach((v, k) => { headers[k] = v; });
    const body = request.body ? new Uint8Array(await request.arrayBuffer()) : EMPTY;
    const meta = { m: request.method, p: url.pathname + url.search, h: headers };
    return concat(cipher.frame(utf8(JSON.stringify(meta)), Q), body.length ? cipher.pack(body, Q) : EMPTY);
  }

  async function wrappedFetch(input, init) {
    const request = new Request(typeof input === "string" || input instanceof URL ? new URL(input, origin).href : input, init), url = new URL(request.url, origin);
    if (!wrapped(url)) return plainFetch(input, init);
    const wire = await pack(request.clone());
    const outer = await plainFetch(ENDPOINT, { method: "POST", body: wire, headers: { "Content-Type": "application/octet-stream" }, signal: request.signal, keepalive: request.keepalive && wire.length < 60000, cache: "no-store", credentials: "same-origin" });
    if (outer.status === 404) { off = true; return plainFetch(input, init); }
    if (outer.status !== 200 || !/octet-stream/.test(outer.headers.get("content-type") || "")) return outer;      // not an envelope answer: the server's own refusal
    return unwrap(outer, cipher);
  }

  /** A form upload with progress (fetch cannot report bytes sent) -> {ok, status, json()} */
  async function upload(path, form, onFraction) {
    const request = new Request(new URL(path, origin), { method: "POST", body: form });
    const url = new URL(request.url);
    if (!wrapped(url) || !plainXhr) return null;
    const wire = await pack(request);
    return new Promise((resolve, reject) => {
      const xhr = plainXhr();
      xhr.open("POST", ENDPOINT);
      xhr.responseType = "arraybuffer";
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onFraction(e.loaded / e.total); };
      xhr.onerror = () => reject(new Error("the upload was interrupted"));
      xhr.onload = () => {
        try {
          const frames = cipher.unpack(new Uint8Array(xhr.response), R), meta = JSON.parse(fromUtf8(frames[0])), text = fromUtf8(concat(...frames.slice(1)));
          resolve({ ok: meta.s >= 200 && meta.s < 300, status: meta.s, statusText: "", json: async () => JSON.parse(text) });
        } catch (err) { reject(err); }
      };
      xhr.send(wire);
    });
  }

  /** The progress socket: frames arrive wrapped (`?e=1` asks for that); the page hears them as it always did. */
  function socket(real) {
    const made = new WeakMap();
    const open = (fn) => {
      if (!made.has(fn)) made.set(fn, (event) => {
        let data = event.data;
        if (typeof data !== "string") {                       // a plain string comes from a server that does not wrap: heard as it is
          let message;
          try { message = cipher.unseal(new Uint8Array(data), W); } catch { return; }
          data = message[0] === 0 ? fromUtf8(message.subarray(1)) : message.slice(1).buffer;
        }
        (typeof fn === "function" ? fn : (e) => fn.handleEvent(e))(new MessageEvent("message", { data }));
      });
      return made.get(fn);
    };
    real.binaryType = "arraybuffer";
    return new Proxy(real, {
      get(target, prop) {
        if (prop === "addEventListener") return (type, fn, opts) => target.addEventListener(type, type === "message" ? open(fn) : fn, opts);
        if (prop === "removeEventListener") return (type, fn, opts) => target.removeEventListener(type, type === "message" ? open(fn) : fn, opts);
        const value = target[prop];
        return typeof value === "function" ? value.bind(target) : value;
      },
      set(target, prop, value) { target[prop] = prop === "onmessage" && value ? open(value) : value; return true; },
    });
  }

  /** The address of something to show (a picture, a clip) -> a blob address of its decrypted bytes, or null when it is not ours to wrap. */
  async function blobFor(href, signal) {
    let url;
    try { url = new URL(href, origin); } catch { return null; }
    if (!wrapped(url)) return null;
    const answer = await wrappedFetch(url.href, { signal });
    if (!answer.ok) return null;
    return URL.createObjectURL(await answer.blob());
  }

  /** The file a download link points at -> { blob, name } */
  async function download(href, name) {
    const url = new URL(href, origin);
    if (!wrapped(url)) return null;
    const answer = await wrappedFetch(url.href);
    if (!answer.ok) return null;
    const disposition = answer.headers.get("content-disposition") || "";
    const given = /filename\*=UTF-8''([^;]+)/i.exec(disposition) || /filename="?([^";]+)"?/i.exec(disposition);
    return { blob: await answer.blob(), name: name || (given ? decodeURIComponent(given[1]) : decodeURIComponent(url.pathname.split("/").pop() || "download")) };
  }

  return { fetch: wrappedFetch, upload, socket, blobFor, download, wrapped, off: () => off };
}

const TAGS = new Set(["IMG", "VIDEO", "AUDIO", "SOURCE"]);

/** Put the envelope under this page: fetch, uploads, the socket, pictures and clips, downloads. Once. */
export function install(win = globalThis) {
  if (win.__funuiEnvelope || !win.fetch) return win.__funuiEnvelope || null;
  const plainFetch = win.fetch.bind(win), Xhr = win.XMLHttpRequest;
  const env = createEnvelope({ plainFetch, origin: win.location.origin, plainXhr: Xhr ? () => new Xhr() : null });
  win.__funuiEnvelope = env;
  win.fetch = env.fetch;

  const revoke = typeof FinalizationRegistry === "function" ? new FinalizationRegistry((href) => URL.revokeObjectURL(href)) : null;
  const latest = new WeakMap();
  const setSrc = (el, href, original) => {
    const mine = {}; latest.set(el, mine);
    const aborter = new AbortController(); mine.abort = aborter;
    env.blobFor(href, aborter.signal).then((blob) => {
      if (latest.get(el) !== mine) { if (blob) URL.revokeObjectURL(blob); return; }
      if (mine.blob) URL.revokeObjectURL(mine.blob);
      mine.blob = blob;
      original(blob || href);
      if (blob && revoke) revoke.register(el, blob);
    }, () => { if (latest.get(el) === mine) original(href); });         // no envelope answer: the plain address, as before
  };
  const wrapsrc = (proto) => {
    const d = Object.getOwnPropertyDescriptor(proto, "src");
    if (!d || !d.set) return;
    Object.defineProperty(proto, "src", { ...d, set(value) {
      const href = String(value);
      const previous = latest.get(this);
      if (previous && previous.abort) previous.abort.abort();
      if (!href || href.startsWith("blob:") || href.startsWith("data:") || !env.wrapped(new URL(href, win.location.href))) { latest.delete(this); return d.set.call(this, value); }
      setSrc(this, href, (v) => d.set.call(this, v));
    } });
  };
  if (win.HTMLImageElement) wrapsrc(win.HTMLImageElement.prototype);
  if (win.HTMLMediaElement) wrapsrc(win.HTMLMediaElement.prototype);
  if (win.HTMLSourceElement) wrapsrc(win.HTMLSourceElement.prototype);
  if (win.Element) {
    const set = win.Element.prototype.setAttribute;
    win.Element.prototype.setAttribute = function (name, value) {
      if (String(name).toLowerCase() === "src" && TAGS.has(this.tagName)) { this.src = value; return; }
      return set.call(this, name, value);
    };
  }
  if (win.HTMLAnchorElement) {
    const click = win.HTMLAnchorElement.prototype.click;
    win.HTMLAnchorElement.prototype.click = function () {
      if (!this.hasAttribute("download") || !this.href || this.href.startsWith("blob:")) return click.call(this);
      env.download(this.href, this.getAttribute("download")).then((got) => {
        if (!got) return click.call(this);
        const a = win.document.createElement("a");
        a.href = URL.createObjectURL(got.blob); a.download = got.name;
        win.document.body.append(a); click.call(a); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      }, () => click.call(this));
    };
  }
  return env;
}

/** The progress socket of this page: wrapped when the envelope is on. */
export function openSocket(url, win = globalThis) {
  const env = win.__funuiEnvelope;
  if (!env || env.off()) return new win.WebSocket(url);
  return env.socket(new win.WebSocket(url + (url.includes("?") ? "&" : "?") + "e=1"));
}
