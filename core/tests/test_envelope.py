"""The envelope's cipher: RFC 8439 vectors, then what a message and a body must do (round trip, tamper, wrong key, cut short)."""

import os

import pytest

from core import envelope as env

KEY = bytes(range(0x80, 0xA0))
NONCE = bytes.fromhex("070000004041424344454647")
AAD = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
PLAIN = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it."
SEALED = bytes.fromhex(
    "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
    "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd0600691")


@pytest.fixture(params=["library", "plain python"])
def cipher(request, monkeypatch):
    if request.param == "plain python":
        monkeypatch.setattr(env, "_Fast", None)
    elif env._Fast is None:
        pytest.skip("the cryptography library is not installed here")


def test_rfc8439_aead_vector(cipher):
    assert env.seal(KEY, NONCE, PLAIN, AAD) == SEALED
    assert env.unseal(KEY, NONCE, SEALED, AAD) == PLAIN


def test_rfc8439_block_and_poly1305_vectors():
    block = env._block(bytes(range(32)), 1, bytes.fromhex("000000090000004a00000000"))
    assert block[:16].hex() == "10f1e7e4d13b5915500fdd1fa32071c4"
    key = bytes.fromhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
    assert env.poly1305(key, b"Cryptographic Forum Research Group").hex() == "a8061dc1305136c6c22b8baf0c0127a9"


def test_the_numpy_keystream_matches_the_plain_one():
    pytest.importorskip("numpy")
    data = os.urandom(64 * 9 + 5)
    plain = b"".join(env._block(KEY, 7 + i, NONCE) for i in range(10))[:len(data)]
    assert env.chacha20(KEY, 7, NONCE, data) == bytes(a ^ b for a, b in zip(data, plain))


def test_a_message_round_trips_and_every_message_has_its_own_nonce(cipher):
    c = env.Cipher()
    one, two = c.seal(b"hello", b"q"), c.seal(b"hello", b"q")
    assert one != two and c.unseal(one, b"q") == c.unseal(two, b"q") == b"hello"
    assert env.Cipher().unseal(one, b"q") == b"hello"                     # the built-in key is the same in every copy


def test_a_changed_message_another_purpose_or_another_key_is_refused(cipher):
    c = env.Cipher()
    sealed = c.seal(b"hello", b"q")
    for bad in (sealed[:-1] + bytes([sealed[-1] ^ 1]), sealed[:14] + bytes([sealed[14] ^ 1]) + sealed[15:], sealed[:-3]):
        with pytest.raises(env.EnvelopeError):
            c.unseal(bad, b"q")
    with pytest.raises(env.EnvelopeError):
        c.unseal(sealed, b"r")                                              # a request message handed over as an answer
    with pytest.raises(env.EnvelopeError):
        env.Cipher(bytes(32)).unseal(sealed, b"q")
    with pytest.raises(env.EnvelopeError):
        c.unseal(b"short")


def test_a_large_body_goes_in_chunks_and_a_cut_one_fails(cipher):
    c = env.Cipher(os.urandom(32))
    body = os.urandom(env.CHUNK * 3 + 123)
    wire = c.pack(body, b"q")
    assert len(c.unpack(wire, b"q")) == 4 and b"".join(c.unpack(wire, b"q")) == body
    with pytest.raises(env.EnvelopeError):
        c.unpack(wire[:len(wire) // 2], b"q")
    assert c.unpack(c.pack(b"", b"q"), b"q") == [b""]
    assert c.unpack(b"", b"q") == []


# ---------------------------------------------------------------- over HTTP

def _server(tmp_path):
    import asyncio
    import json
    from aiohttp import web
    from core import envelope_http as eh
    routes = web.RouteTableDef()
    (tmp_path / "pic.png").write_bytes(b"\x89PNG" + os.urandom(150000))

    @routes.post("/fun/api/echo")
    async def _echo(req):
        return web.json_response({"got": await req.json(), "q": req.query.get("a"), "ct": req.content_type})

    @routes.post("/upload/image")
    async def _upload(req):
        form = await req.post()
        return web.json_response({"name": form["image"].filename, "size": len(form["image"].file.read()), "kind": form["type"]})

    @routes.get("/view")
    async def _view(req):
        return web.FileResponse(tmp_path / "pic.png", headers={"Content-Type": "image/png"})

    @routes.get("/fun/api/lines")
    async def _lines(req):
        resp = eh.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for i in range(3):
            await resp.write(f"data: {i}\n\n".encode())
        await resp.write_eof()
        return resp

    @routes.get("/fun/api/plain")
    async def _plain(req):
        return web.Response(text="hello", content_type="text/plain")

    app = web.Application()
    eh.register(routes)
    app.add_routes(routes)
    return app, eh, json, asyncio


def _run(tmp_path, scenario):
    from aiohttp.test_utils import TestClient, TestServer
    app, eh, json, asyncio = _server(tmp_path)

    async def go():
        async with TestClient(TestServer(app)) as client:
            async def call(method, path, body=b"", ctype="", headers=None):
                meta = {"m": method, "p": path, "h": {**({"Content-Type": ctype} if ctype else {}), **(headers or {})}}
                wire = eh.CIPHER.frame(json.dumps(meta).encode(), eh.REQUEST) + (eh.CIPHER.pack(body, eh.REQUEST) if body else b"")
                resp = await client.post("/fun/x", data=wire)
                raw = await resp.read()
                assert resp.status == 200 and resp.content_type == "application/octet-stream"
                frames = eh.CIPHER.unpack(raw, eh.ANSWER)
                head = json.loads(frames[0])
                return head["s"], head["h"], b"".join(frames[1:]), raw
            return await scenario(call, client, eh)
    return asyncio.run(go())


def test_calls_through_the_one_address_reach_the_real_routes_and_come_back_wrapped(tmp_path):
    async def scenario(call, client, eh):
        status, headers, body, raw = await call("POST", "/fun/api/echo?a=7", b'{"prompt": "secret words"}', "application/json")
        assert status == 200 and json_of(body) == {"got": {"prompt": "secret words"}, "q": "7", "ct": "application/json"}
        assert b"secret" not in raw and b"prompt" not in raw
        status, headers, body, _ = await call("GET", "/fun/api/plain")
        assert (status, body, headers["Content-Type"].startswith("text/plain")) == (200, b"hello", True)
        status, _, body, _ = await call("GET", "/fun/api/nowhere")
        assert status == 404
        assert (await client.post("/fun/x", data=b"not an envelope")).status == 400
        assert (await client.get("/fun/api/plain")).status == 200                    # plain addresses keep working
    json_of = __import__("json").loads
    _run(tmp_path, scenario)


def test_a_file_upload_a_finished_picture_and_a_stream_survive_the_envelope(tmp_path):
    async def scenario(call, client, eh):
        form = (b"--B\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n" + os.urandom(100000)
                + b"\r\n--B\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--B--\r\n")
        status, _, body, raw = await call("POST", "/upload/image", form, "multipart/form-data; boundary=B")
        assert json.loads(body) == {"name": "a.png", "size": 100000, "kind": "input"}
        status, headers, body, raw = await call("GET", "/view?filename=pic.png")
        assert body == (tmp_path / "pic.png").read_bytes() and headers["Content-Type"] == "image/png" and b"PNG" not in raw[:200]
        status, headers, body, raw = await call("GET", "/fun/api/lines")
        assert body == b"data: 0\n\ndata: 1\n\ndata: 2\n\n" and headers["Content-Type"] == "text/event-stream" and b"data:" not in raw
        plain = await client.get("/fun/api/lines")                                    # the same handler, asked plainly, writes plainly
        assert (await plain.read()) == b"data: 0\n\ndata: 1\n\ndata: 2\n\n"
    import json
    _run(tmp_path, scenario)
