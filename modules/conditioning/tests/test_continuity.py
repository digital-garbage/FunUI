from core import registry


def test_continuity_is_announced_in_its_own_category_with_its_choices():
    spec = registry.scan().specs["continuity"]
    assert spec.category == "continuity"
    assert set(spec.settings) == {"carry", "tail_video", "tail_audio", "tail_seconds", "tail_frames"}
    assert spec.settings["carry"]["default"] is True


def test_continuity_is_h3_only_for_now():
    spec = registry.scan().specs["continuity"]
    assert "frame_carry" in spec.requires


def test_the_end_of_the_clip_is_off_by_default_and_seconds_and_frames_agree_at_25_fps():
    spec = registry.scan().specs["continuity"]
    assert spec.settings["tail_video"]["default"] is False and spec.settings["tail_audio"]["default"] is False
    assert spec.settings["tail_seconds"]["default"] * 25 == spec.settings["tail_frames"]["default"]
    assert spec.settings["tail_seconds"]["sync"]["with"] == "tail_frames" and spec.settings["tail_frames"]["sync"]["with"] == "tail_seconds"
