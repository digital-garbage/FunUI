import asyncio
import io

import pytest
from PIL import Image

from core import config, media
from modules.media import refsheet
from modules.media.refsheet import sheet


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")


def _png(size, color):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "PNG")
    return out.getvalue()


def _add(size, color, name="p.png"):
    return media.save_upload(name, _png(size, color))["id"]


def _decode(png):
    return Image.open(io.BytesIO(png)).convert("RGB")


def test_every_picture_is_cut_to_the_slot_and_the_rest_is_white(store):
    wide, tall = _add((400, 100), (255, 0, 0)), _add((100, 400), (0, 0, 255))
    png = sheet.make(1, 2, "1:1", 256, "", [{"picture": wide}, {"picture": tall}], refsheet._open)
    im = _decode(png)
    w = 256
    gap, margin = round(w * 0.06), round(w * 0.08)
    assert im.size == (margin * 2 + 2 * w + gap, margin * 2 + w + round(max(12, round(w * 0.055)) * 1.9))
    assert im.getpixel((0, 0)) == (255, 255, 255)
    assert im.getpixel((margin + w // 2, margin + w // 2)) == (255, 0, 0)         # the middle of a wide picture fills the whole square
    assert im.getpixel((margin + 2, margin + 2)) == (255, 0, 0) and im.getpixel((margin + w - 3, margin + w - 3)) == (255, 0, 0)
    assert im.getpixel((margin + w + gap + w // 2, margin + w // 2)) == (0, 0, 255)


def _ink(im, box):
    return any(sum(im.getpixel((x, y))) < 200 for x in range(box[0], box[2]) for y in range(box[1], box[3]))


def test_a_label_sits_below_its_slot_even_when_the_slot_is_empty(store):
    pic = _add((64, 64), (0, 200, 0))
    plain = _decode(sheet.make(1, 2, "1:1", 256, "", [{"picture": pic}, {}], refsheet._open))
    drawn = _decode(sheet.make(1, 2, "1:1", 256, "", [{"picture": pic, "label": "front"}, {"label": "back"}], refsheet._open))
    margin, w, gap = 20, 256, 15
    for x in (margin, margin + w + gap):                                    # both slots' label strips
        strip = (x, margin + w + 2, x + w, drawn.size[1])
        assert not _ink(plain, strip) and _ink(drawn, strip)


def test_the_title_is_in_the_top_left_corner(store):
    im = _decode(sheet.make(1, 1, "1:1", 256, "Hero", [], refsheet._open))
    assert _ink(im, (20, 20, 200, 60)) and not _ink(im, (im.size[0] // 2, 0, im.size[0], 60))


def test_a_long_label_is_cut_to_the_slot_width(store):
    from PIL import ImageDraw
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    text, font = sheet._fit(d, "word " * 60, 28, 200)
    assert text.endswith("…") and d.textlength(text, font=font) <= 200


@pytest.mark.parametrize("bad", [
    dict(rows=0), dict(cols=13), dict(rows=True), dict(width=64), dict(width=2000), dict(shape="5:4"), dict(rows=12, cols=12),
])
def test_a_bad_request_is_refused_by_name(bad):
    args = dict(rows=2, cols=2, shape="1:1", width=512, slots=[]) | bad
    with pytest.raises(ValueError):
        sheet.check(args["rows"], args["cols"], args["shape"], args["width"], args["slots"])


def test_too_many_slots_and_a_huge_sheet_are_refused():
    with pytest.raises(ValueError, match="at most 4"):
        sheet.check(2, 2, "1:1", 512, [{}] * 5)
    with pytest.raises(ValueError, match="pixels"):
        sheet.make(6, 6, "9:16", 1024, "", [], lambda m: None)


def test_the_route_saves_the_sheet_to_the_bin_and_names_pictures_it_could_not_read(store):
    ok = _add((50, 50), (9, 9, 9))
    table = {}

    class T:
        def post(self, path): return lambda fn: table.setdefault(path, fn)
        def get(self, path): return lambda fn: table.setdefault(path, fn)

    class W:
        @staticmethod
        def json_response(d, status=200): return {"status": status, **d}
    refsheet.routes(T(), "/m", W)

    def ask(body):
        class R:
            async def json(self): return body
        return asyncio.run(table["/m/make"](R()))

    got = ask({"rows": 1, "cols": 2, "format": "4:3", "width": 256, "title": "Mara", "slots": [{"picture": ok, "label": "front"}, {"picture": "ffffffffffff", "label": "gone"}]})
    assert got["status"] == 200 and got["missing"] == ["gone"]
    item = media.get(got["media"]["id"])
    assert item["kind"] == "image" and item["name"] == "Mara.png" and _decode(media.path_for(item["id"]).read_bytes()).size[0] > 512
    assert ask({"rows": 99})["status"] == 400 and ask("nope")["status"] == 400
    assert [m["id"] for m in media.listing()].count(item["id"]) == 1


def test_a_video_or_sound_in_a_slot_is_an_empty_slot(store):
    vid = media.save_upload("clip.mp4", b"not really a video")["id"]
    assert refsheet._open(vid) is None
