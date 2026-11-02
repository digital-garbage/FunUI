from core import registry


def test_continuity_is_announced_in_its_own_category_with_the_two_choices():
    spec = registry.scan().specs["continuity"]
    assert spec.category == "continuity"
    assert set(spec.settings) == {"carry", "dark_guard"}
    assert spec.settings["carry"]["default"] is True and spec.settings["dark_guard"]["default"] is True


def test_continuity_is_h3_only_for_now():
    spec = registry.scan().specs["continuity"]
    assert "frame_carry" in spec.requires
