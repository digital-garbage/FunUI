"""The light-privacy envelope: ChaCha20-Poly1305 (RFC 8439) between the browser and the backend.

Self-contained (standard library only, no ComfyUI, no web framework) so the standalone app can reuse it, and the key is
injected: FunUI uses a fixed built-in one (`DEFAULT_KEY`), a stronger key source can be passed to `Cipher` later. The browser
half is app/ui/shell/envelope_cipher.js; both are tested against the RFC 8439 vectors.

This is LIGHT PRIVACY, not security: the key is the same in every copy of FunUI, so anyone who knows the method can read the
traffic. It keeps what is asked and generated from being readable at a glance by someone watching the wire (a shared network,
a rental provider's logs). It does not hide that a generation happens, and it does not defend against an attacker.

A message is `nonce (12, random) | ciphertext | tag (16)`. A body travels as length-prefixed messages of at most 64 KiB, so a
large file streams without being held whole.
"""

import hashlib
import hmac
import os
import struct

TAG = 16
NONCE = 12
CHUNK = 64 * 1024
_MASK = 0xFFFFFFFF


class EnvelopeError(Exception):
    """A message that does not belong: another key, changed, or cut short."""


# ---------------------------------------------------------------- ChaCha20-Poly1305 (RFC 8439)

try:                                            # the same cipher, in C, when the machine has it
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305 as _Fast
except Exception:                               # noqa: BLE001
    _Fast = None


def _rotl(v, n):
    return ((v << n) & _MASK) | (v >> (32 - n))


def _block(key, counter, nonce):
    state = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *struct.unpack("<8I", key), counter & _MASK, *struct.unpack("<3I", nonce)]
    x = list(state)

    def quarter(a, b, c, d):
        x[a] = (x[a] + x[b]) & _MASK; x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & _MASK; x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & _MASK; x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & _MASK; x[b] = _rotl(x[b] ^ x[c], 7)
    for _ in range(10):
        quarter(0, 4, 8, 12); quarter(1, 5, 9, 13); quarter(2, 6, 10, 14); quarter(3, 7, 11, 15)
        quarter(0, 5, 10, 15); quarter(1, 6, 11, 12); quarter(2, 7, 8, 13); quarter(3, 4, 9, 14)
    return struct.pack("<16I", *[(x[i] + state[i]) & _MASK for i in range(16)])


def _keystream_numpy(key, counter, nonce, blocks):
    import numpy as np
    base = np.array([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *struct.unpack("<8I", key), 0, *struct.unpack("<3I", nonce)], dtype=np.uint32)
    state = np.tile(base, (blocks, 1))
    state[:, 12] = (np.arange(blocks, dtype=np.uint64) + counter).astype(np.uint32)
    x = state.copy()

    def rot(v, n):
        return (v << np.uint32(n)) | (v >> np.uint32(32 - n))

    def quarter(a, b, c, d):
        x[:, a] += x[:, b]; x[:, d] = rot(x[:, d] ^ x[:, a], 16)
        x[:, c] += x[:, d]; x[:, b] = rot(x[:, b] ^ x[:, c], 12)
        x[:, a] += x[:, b]; x[:, d] = rot(x[:, d] ^ x[:, a], 8)
        x[:, c] += x[:, d]; x[:, b] = rot(x[:, b] ^ x[:, c], 7)
    with np.errstate(over="ignore"):
        for _ in range(10):
            quarter(0, 4, 8, 12); quarter(1, 5, 9, 13); quarter(2, 6, 10, 14); quarter(3, 7, 11, 15)
            quarter(0, 5, 10, 15); quarter(1, 6, 11, 12); quarter(2, 7, 8, 13); quarter(3, 4, 9, 14)
        x += state
    return x.astype("<u4").tobytes()


def chacha20(key, counter, nonce, data):
    """The RFC 8439 stream cipher (encrypt and decrypt are the same)."""
    blocks = (len(data) + 63) // 64
    if not blocks:
        return b""
    try:
        stream = _keystream_numpy(key, counter, nonce, blocks) if blocks > 4 else None
    except ImportError:
        stream = None
    if stream is None:
        stream = b"".join(_block(key, counter + i, nonce) for i in range(blocks))
    n = len(data)
    return (int.from_bytes(data, "little") ^ int.from_bytes(stream[:n], "little")).to_bytes(n, "little")


def poly1305(key, message):
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:32], "little")
    p, acc = (1 << 130) - 5, 0
    for i in range(0, len(message), 16):
        block = message[i:i + 16]
        acc = (acc + int.from_bytes(block + b"\x01", "little")) * r % p
    return ((acc + s) & ((1 << 128) - 1)).to_bytes(16, "little")


def _pad(data):
    return b"\x00" * (-len(data) % 16)


def _mac_data(aad, ciphertext):
    return aad + _pad(aad) + ciphertext + _pad(ciphertext) + struct.pack("<QQ", len(aad), len(ciphertext))


def seal(key, nonce, plaintext, aad=b""):
    """ciphertext and 16-byte tag."""
    if _Fast is not None:
        return _Fast(key).encrypt(nonce, plaintext, aad)
    ciphertext = chacha20(key, 1, nonce, plaintext)
    return ciphertext + poly1305(_block(key, 0, nonce)[:32], _mac_data(aad, ciphertext))


def unseal(key, nonce, sealed, aad=b""):
    """The plaintext, or EnvelopeError when the key, the frame or what it belongs to differs."""
    if len(sealed) < TAG:
        raise EnvelopeError("a frame is shorter than its tag")
    if _Fast is not None:
        try:
            return _Fast(key).decrypt(nonce, bytes(sealed), aad)
        except Exception:                       # noqa: BLE001
            raise EnvelopeError("a frame did not verify: wrong code, or the data was changed") from None
    ciphertext, tag = sealed[:-TAG], sealed[-TAG:]
    if not hmac.compare_digest(poly1305(_block(key, 0, nonce)[:32], _mac_data(aad, ciphertext)), tag):
        raise EnvelopeError("a frame did not verify: wrong code, or the data was changed")
    return chacha20(key, 1, nonce, ciphertext)


# ---------------------------------------------------------------- messages and bodies

#: The key every copy of FunUI shares (light privacy: see the top of this file).
DEFAULT_KEY = hashlib.sha256(b"FunUI light privacy v1").digest()


class Cipher:
    def __init__(self, key=DEFAULT_KEY):
        if len(key) != 32:
            raise ValueError("the key is 32 bytes")
        self.key = bytes(key)

    def seal(self, data, purpose=b""):
        nonce = os.urandom(NONCE)
        return nonce + seal(self.key, nonce, bytes(data), purpose)

    def unseal(self, message, purpose=b""):
        if len(message) < NONCE + TAG:
            raise EnvelopeError("a message is cut short")
        return unseal(self.key, message[:NONCE], message[NONCE:], purpose)

    def pack(self, data, purpose=b""):
        """A body -> length-prefixed messages (at least one, so an empty body still has its marker)."""
        data = bytes(data)
        pieces = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [b""]
        return b"".join(self.frame(p, purpose) for p in pieces)

    def frame(self, data, purpose=b""):
        message = self.seal(data, purpose)
        return struct.pack(">I", len(message)) + message

    def unpack(self, data, purpose=b""):
        """Length-prefixed messages -> their contents, in order."""
        data, at, out = bytes(data), 0, []
        while at < len(data):
            if at + 4 > len(data):
                raise EnvelopeError("a message is cut short")
            size = struct.unpack(">I", data[at:at + 4])[0]
            if size > CHUNK + NONCE + TAG or at + 4 + size > len(data):
                raise EnvelopeError("a message is cut short")
            out.append(self.unseal(data[at + 4:at + 4 + size], purpose))
            at += 4 + size
        return out
