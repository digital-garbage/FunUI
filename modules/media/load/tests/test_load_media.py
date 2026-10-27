"""FunUILoadMedia: the bridge from a media library id to an IMAGE tensor."""

import io as pyio

import pytest

from core import config, media


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy_api."""


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    return tmp_path / "media"


def _png_bytes():
    from PIL import Image
    buf = pyio.BytesIO()
    Image.new("RGB", (4, 6), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_a_picked_image_loads_as_an_image_tensor(store):
    from modules.media.load.nodes import FunUILoadMedia
    entry = media.save_upload("ref.png", _png_bytes())

    images, frames, _audio = FunUILoadMedia.execute(entry["id"]).result
    assert images.shape[0] == 1 and frames.shape[0] == 1
    assert images.shape[1:3] == (6, 4)   # H, W


def test_no_media_id_is_refused_loudly(store):
    from modules.media.load.nodes import FunUILoadMedia
    with pytest.raises(RuntimeError, match="nothing is picked"):
        FunUILoadMedia.execute("").result


def test_an_unknown_id_is_refused_loudly(store):
    from modules.media.load.nodes import FunUILoadMedia
    with pytest.raises(RuntimeError, match="not in the library"):
        FunUILoadMedia.execute("deadbeef0000").result


def test_a_file_that_cannot_be_read_is_refused_in_plain_words(store):
    from modules.media.load.nodes import FunUILoadMedia
    for name in ("clip.mp4", "voice.wav"):
        entry = media.save_upload(name, b"not real media but has an ext")
        with pytest.raises(RuntimeError, match="could not read"):
            FunUILoadMedia.execute(entry["id"]).result


def test_deleted_media_is_refused_even_with_a_well_formed_id(store):
    from modules.media.load.nodes import FunUILoadMedia
    entry = media.save_upload("ref.png", _png_bytes())
    media.delete(entry["id"])
    with pytest.raises(RuntimeError, match="not in the library"):
        FunUILoadMedia.execute(entry["id"]).result


def test_a_blank_size_stands_in_for_nothing_picked_and_a_picked_image_still_wins(store):
    from modules.media.load.nodes import FunUILoadMedia
    images, *_ = FunUILoadMedia.execute("", blank_width=64, blank_height=32).result
    assert tuple(images.shape) == (1, 32, 64, 3)
    entry = media.save_upload("ref.png", _png_bytes())
    images, *_ = FunUILoadMedia.execute(entry["id"], blank_width=64, blank_height=32).result
    assert images.shape[1:3] == (6, 4)
    with pytest.raises(RuntimeError, match="nothing is picked"):
        FunUILoadMedia.execute("", blank_width=64, blank_height=0).result


def _clip(store, seconds=1):
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed to make a clip")
    clip = store.parent / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=8:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-pix_fmt", "yuv420p", str(clip)], check=True)
    return media.save_upload("clip.mp4", clip.read_bytes())


def test_a_video_gives_its_first_frame_by_default(store):
    from modules.media.load.nodes import FunUILoadMedia
    entry = _clip(store)
    image, frames, _ = FunUILoadMedia.execute(entry["id"]).result
    assert image.shape[0] == 1 and frames.shape[0] == 1
    assert image.shape[1:3] == (48, 64)


def test_a_full_video_gives_every_frame_at_24_fps_and_its_sound(store):
    from modules.media.load.nodes import FunUILoadMedia
    entry = _clip(store, seconds=2)
    image, frames, audio = FunUILoadMedia.execute(entry["id"], full=True).result
    assert image.shape[0] == 1
    assert frames.shape[0] == 48            # 2 s at 24 fps
    assert audio["waveform"].shape[-1] > 1000


def test_audio_loads_as_audio(store):
    from modules.media.load.nodes import FunUILoadMedia
    import struct, wave
    path = store.parent / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(struct.pack("<800h", *([1000] * 800)))
    entry = media.save_upload("a.wav", path.read_bytes())
    _, _, audio = FunUILoadMedia.execute(entry["id"], full=True).result
    assert audio["sample_rate"] == 8000 and audio["waveform"].shape == (1, 1, 800)


def test_slots_of_the_retired_video_and_audio_nodes_are_dropped_with_their_wires():
    from modules.media.load import upgrade
    slots = [{"id": "r2v", "node": "X", "inputs": {"a": ["v1", 1], "keep": 3, "b": ["ref", 0]}},
             {"id": "v1", "node": "FunUILoadMediaVideo", "inputs": {}},
             {"id": "ref", "node": "FunUILoadMedia", "inputs": {}}]
    out = upgrade(slots, None)
    assert [s["id"] for s in out] == ["r2v", "ref"]
    assert out[0]["inputs"] == {"keep": 3, "b": ["ref", 0]}


def _long_clip(store, frames=30, rate=30, size=(64, 48)):
    import av
    import numpy as np
    store.parent.mkdir(parents=True, exist_ok=True)
    path = store.parent / "long.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(frames):
            arr = np.full((size[1], size[0], 3), (i * 7) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return media.save_upload("long.mp4", path.read_bytes())["id"]


def test_a_prepared_video_decodes_only_what_it_keeps_and_matches_the_whole_decode_route(store):
    import json
    import torch
    from comfy_api.latest import InputImpl
    from modules.media.load import prepare as prep
    from modules.media.load.nodes import FunUILoadMedia
    mid = _long_clip(store)
    options = {"fps": 24, "skip_first": 2, "cut_last": 3, "every_nth": 2, "frame_cap": 6, "width": 32, "height": 32, "fit": "crop", "multiple_of": 16}

    _, frames, _ = FunUILoadMedia.execute(mid, full=True, prepare=json.dumps(options)).result

    comp = InputImpl.VideoFromFile(str(media.path_for(mid))).get_components()
    expected, _, _, _ = prep.prepare(comp.images, float(comp.frame_rate), options)
    assert frames.shape == expected.shape and frames.shape[0] == 6
    assert torch.allclose(frames, expected, atol=1 / 255)


def test_a_cut_that_leaves_nothing_is_explained_and_a_picture_input_reads_one_frame(store):
    import json
    from modules.media.load.nodes import FunUILoadMedia
    mid = _long_clip(store)
    with pytest.raises(RuntimeError, match="cannot be prepared"):
        FunUILoadMedia.execute(mid, full=True, prepare=json.dumps({"skip_first": 100})).result
    image, _, _ = FunUILoadMedia.execute(mid).result
    assert image.shape == (1, 48, 64, 3)
