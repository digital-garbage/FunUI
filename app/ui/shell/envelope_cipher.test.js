import test from "node:test";
import assert from "node:assert/strict";
import * as env from "./envelope_cipher.js";

const hex = (s) => Uint8Array.from(s.match(/../g).map((h) => parseInt(h, 16)));
const toHex = (b) => [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
const bytes = (n, f) => Uint8Array.from({ length: n }, (_, i) => f(i));

test("RFC 8439: the AEAD vector, the block function and Poly1305", () => {
  const key = bytes(32, (i) => 0x80 + i), nonce = hex("070000004041424344454647"), aad = hex("50515253c0c1c2c3c4c5c6c7");
  const plain = env.utf8("Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.");
  const sealed = env.seal(key, nonce, plain, aad);
  assert.equal(toHex(sealed), "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd0600691");
  assert.deepEqual(env.unseal(key, nonce, sealed, aad), plain);
  assert.equal(toHex(env.chacha20(bytes(32, (i) => i), 1, hex("000000090000004a00000000"), new Uint8Array(16))), "10f1e7e4d13b5915500fdd1fa32071c4");
  assert.equal(toHex(env.poly1305(hex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b"), env.utf8("Cryptographic Forum Research Group"))), "a8061dc1305136c6c22b8baf0c0127a9");
});

test("SHA-256 (the built-in key is derived from it)", () => {
  assert.equal(toHex(env.sha256(env.utf8("abc"))), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert.equal(toHex(env.sha256(env.utf8("a".repeat(1000)))), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
});

test("a message round trips with its own nonce, and the built-in key is the Python half's", () => {
  const c = new env.Cipher(), a = c.seal(env.utf8("hello"), env.utf8("q")), b = c.seal(env.utf8("hello"), env.utf8("q"));
  assert.notDeepEqual(a, b);
  assert.equal(env.fromUtf8(c.unseal(a, env.utf8("q"))), "hello");
  assert.equal(toHex(env.DEFAULT_KEY), "73bfe9fa78668f9a7acdb253389857a18d74297de1871241bd247767415240d3");
  assert.equal(env.fromUtf8(c.unseal(hex("e48121f175f1d6d685f1769f41ac9c5fe01a83cb320c391cf4bf7f977a8a84560bfadad9dedfad3ab53b08d8b4"), env.utf8("q"))), "hello from python");
});

test("a changed message, another purpose or key, and a cut one are refused; a large body goes in chunks", () => {
  const c = new env.Cipher(env.randomBytes(32)), q = env.utf8("q");
  const sealed = c.seal(env.utf8("hello"), q), flipped = sealed.slice(); flipped[20] ^= 1;
  assert.throws(() => c.unseal(flipped, q), env.EnvelopeError);
  assert.throws(() => c.unseal(sealed, env.utf8("r")), env.EnvelopeError);
  assert.throws(() => new env.Cipher(env.randomBytes(32)).unseal(sealed, q), env.EnvelopeError);
  assert.throws(() => c.unseal(sealed.subarray(0, 10), q), env.EnvelopeError);
  const body = env.randomBytes(env.CHUNK * 3 + 123), wire = c.pack(body, q), parts = c.unpack(wire, q);
  assert.equal(parts.length, 4);
  assert.deepEqual(env.concat(...parts), body);
  assert.throws(() => c.unpack(wire.subarray(0, wire.length >> 1), q), env.EnvelopeError);
  assert.deepEqual(c.unpack(c.pack(new Uint8Array(0), q), q).map((p) => p.length), [0]);
});
