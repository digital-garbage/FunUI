// The light-privacy envelope, browser half: ChaCha20-Poly1305 (RFC 8439) with no WebCrypto.
//
// Browsers only offer crypto.subtle on https and localhost, and a rented machine is plain http by IP, so the cipher is
// written out here. Standard library only, no imports, so the standalone app can reuse it; the key is injected
// (`new Cipher(key)`), FunUI's own is the fixed built-in one. Twin of core/envelope.py, tested against the same RFC 8439 vectors.
//
// LIGHT PRIVACY, not security: every copy of FunUI has the same key, so anyone who knows the method can read the traffic. It
// keeps what is asked and generated from being readable at a glance on the wire. It does not hide that a generation happens.
//
// A message is nonce (12, random) | ciphertext | tag (16); a body travels as length-prefixed messages of at most 64 KiB.

export const TAG = 16, NONCE = 12, CHUNK = 64 * 1024;
const text = new TextEncoder();

export class EnvelopeError extends Error {}

const cat = (...parts) => { const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0)); let at = 0; for (const p of parts) { out.set(p, at); at += p.length; } return out; };
const same = (a, b) => { if (a.length !== b.length) return false; let d = 0; for (let i = 0; i < a.length; i++) d |= a[i] ^ b[i]; return d === 0; };

// ---------------------------------------------------------------- SHA-256, HMAC, HKDF

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);

export function sha256(data) {
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const padded = new Uint8Array(((data.length + 9 + 63) >> 6) << 6);
  padded.set(data); padded[data.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(data.length / 0x20000000));
  view.setUint32(padded.length - 4, (data.length << 3) >>> 0);
  const w = new Uint32Array(64);
  for (let at = 0; at < padded.length; at += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(at + i * 4);
    for (let i = 16; i < 64; i++) {
      const a = w[i - 15], b = w[i - 2];
      w[i] = (w[i - 16] + (((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3)) + w[i - 7] + (((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10))) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const t1 = (hh + (((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7))) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0;
      const t2 = ((((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10))) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  const out = new Uint8Array(32), o = new DataView(out.buffer);
  h.forEach((v, i) => o.setUint32(i * 4, v));
  return out;
}

// ---------------------------------------------------------------- ChaCha20-Poly1305 (RFC 8439)

const rotl = (v, n) => (v << n) | (v >>> (32 - n));

function chachaBlock(key, counter, nonce, out, at) {
  const k = new DataView(key.buffer, key.byteOffset, 32), n = new DataView(nonce.buffer, nonce.byteOffset, 12);
  const s = new Uint32Array(16);
  s.set([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]);
  for (let i = 0; i < 8; i++) s[4 + i] = k.getUint32(i * 4, true);
  s[12] = counter >>> 0;
  for (let i = 0; i < 3; i++) s[13 + i] = n.getUint32(i * 4, true);
  const x = Uint32Array.from(s);
  const q = (a, b, c, d) => {
    x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 12);
    x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 8); x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 7);
  };
  for (let i = 0; i < 10; i++) { q(0, 4, 8, 12); q(1, 5, 9, 13); q(2, 6, 10, 14); q(3, 7, 11, 15); q(0, 5, 10, 15); q(1, 6, 11, 12); q(2, 7, 8, 13); q(3, 4, 9, 14); }
  const o = new DataView(out.buffer, out.byteOffset + at, 64);
  for (let i = 0; i < 16; i++) o.setUint32(i * 4, (x[i] + s[i]) >>> 0, true);
}

export function chacha20(key, counter, nonce, data) {
  const out = new Uint8Array(data.length), block = new Uint8Array(64);
  for (let at = 0, c = counter; at < data.length; at += 64, c++) {
    chachaBlock(key, c, nonce, block, 0);
    const end = Math.min(64, data.length - at);
    for (let i = 0; i < end; i++) out[at + i] = data[at + i] ^ block[i];
  }
  return out;
}

const P = (1n << 130n) - 5n;
const le = (bytes) => { let v = 0n; for (let i = bytes.length - 1; i >= 0; i--) v = (v << 8n) | BigInt(bytes[i]); return v; };

export function poly1305(key, message) {
  const r = le(key.subarray(0, 16)) & 0x0ffffffc0ffffffc0ffffffc0fffffffn, s = le(key.subarray(16, 32));
  let acc = 0n;
  for (let i = 0; i < message.length; i += 16) {
    const part = message.subarray(i, i + 16);
    acc = ((acc + (le(part) | (1n << BigInt(8 * part.length)))) * r) % P;
  }
  let v = (acc + s) & ((1n << 128n) - 1n);
  const out = new Uint8Array(16);
  for (let i = 0; i < 16; i++) { out[i] = Number(v & 0xffn); v >>= 8n; }
  return out;
}

const padded = (a) => cat(a, new Uint8Array((16 - (a.length % 16)) % 16));
function macData(aad, ciphertext) {
  const lens = new DataView(new ArrayBuffer(16));
  lens.setBigUint64(0, BigInt(aad.length), true); lens.setBigUint64(8, BigInt(ciphertext.length), true);
  return cat(padded(aad), padded(ciphertext), new Uint8Array(lens.buffer));
}
const polyKey = (key, nonce) => { const b = new Uint8Array(64); chachaBlock(key, 0, nonce, b, 0); return b.subarray(0, 32); };

/** ciphertext followed by the 16-byte tag */
export function seal(key, nonce, plaintext, aad = new Uint8Array(0)) {
  const ciphertext = chacha20(key, 1, nonce, plaintext);
  return cat(ciphertext, poly1305(polyKey(key, nonce), macData(aad, ciphertext)));
}

/** the plaintext, or an EnvelopeError when the key, the frame or what it belongs to differs */
export function unseal(key, nonce, sealed, aad = new Uint8Array(0)) {
  if (sealed.length < TAG) throw new EnvelopeError("a frame is shorter than its tag");
  const ciphertext = sealed.subarray(0, sealed.length - TAG), tag = sealed.subarray(sealed.length - TAG);
  if (!same(poly1305(polyKey(key, nonce), macData(aad, ciphertext)), tag)) throw new EnvelopeError("a frame did not verify: wrong code, or the data was changed");
  return chacha20(key, 1, nonce, ciphertext);
}

// ---------------------------------------------------------------- messages and bodies

export function randomBytes(n) {
  if (!globalThis.crypto || !globalThis.crypto.getRandomValues) throw new EnvelopeError("this browser has no random numbers");      // available on plain http too (only crypto.subtle is not)
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i += 65536) globalThis.crypto.getRandomValues(out.subarray(i, i + 65536));      // at most 65536 bytes per call
  return out;
}

/** The key every copy of FunUI shares (light privacy: see the top of this file). */
export const DEFAULT_KEY = sha256(text.encode("FunUI light privacy v1"));

export class Cipher {
  constructor(key = DEFAULT_KEY) { if (key.length !== 32) throw new EnvelopeError("the key is 32 bytes"); this.key = key; }
  seal(data, purpose = new Uint8Array(0)) { const nonce = randomBytes(NONCE); return cat(nonce, seal(this.key, nonce, data, purpose)); }
  unseal(message, purpose = new Uint8Array(0)) {
    if (message.length < NONCE + TAG) throw new EnvelopeError("a message is cut short");
    return unseal(this.key, message.subarray(0, NONCE), message.subarray(NONCE), purpose);
  }
  /** one message, with its length in front */
  frame(data, purpose) { const m = this.seal(data, purpose), len = new DataView(new ArrayBuffer(4)); len.setUint32(0, m.length); return cat(new Uint8Array(len.buffer), m); }
  /** a body -> length-prefixed messages (at least one, so an empty body still has its marker) */
  pack(data, purpose) {
    const pieces = [];
    for (let i = 0; i < data.length; i += CHUNK) pieces.push(data.subarray(i, i + CHUNK));
    if (!pieces.length) pieces.push(new Uint8Array(0));
    return cat(...pieces.map((p) => this.frame(p, purpose)));
  }
  /** length-prefixed messages -> their contents, in order */
  unpack(data, purpose) {
    const out = [], view = new DataView(data.buffer, data.byteOffset, data.length);
    for (let at = 0; at < data.length;) {
      if (at + 4 > data.length) throw new EnvelopeError("a message is cut short");
      const size = view.getUint32(at);
      if (size > CHUNK + NONCE + TAG || at + 4 + size > data.length) throw new EnvelopeError("a message is cut short");
      out.push(this.unseal(data.subarray(at + 4, at + 4 + size), purpose));
      at += 4 + size;
    }
    return out;
  }
}

export const concat = cat;
export const utf8 = (s) => text.encode(s);
export const fromUtf8 = (b) => new TextDecoder().decode(b);
