"""Find, find-and-replace and common phrases across the whole shortcut library."""

import pytest

from core import config, shortcuts


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SHORTCUTS_FILE", tmp_path / "shortcuts.json")
    monkeypatch.setattr(config, "SHORTCUT_CATEGORIES_FILE", tmp_path / "cats.json")
    monkeypatch.setattr(config, "REVOLVER_FILE", tmp_path / "rev.json")


def put(name, *replacements, enabled=True):
    shortcuts.save({"name": name, "triggers": [name.lower()], "replacements": list(replacements), "enabled": enabled})


def texts():
    return {s.name: s.replacements for s in shortcuts.listing()}


def test_find_lists_every_hit_with_its_count_and_ignores_case_by_default(store):
    put("a", "Red fox, red RED hat", "none")
    put("b", "a red door")
    got = shortcuts.find_replace("red")["matches"]
    assert [(m["name"], m["index"], m["count"]) for m in got] == [("a", 0, 3), ("b", 0, 1)]
    assert shortcuts.find_replace("red", match_case=True)["matches"][0]["count"] == 1   # only the lower-case one


def test_nothing_is_written_by_find_or_by_a_preview(store):
    put("a", "red fox")
    shortcuts.find_replace("red")
    shortcuts.find_replace("red", "blue")
    assert texts() == {"a": ["red fox"]}


def test_apply_replaces_everywhere_and_saves_including_paused_shortcuts(store):
    put("a", "red fox", "a Red hat")
    put("b", "dark red door", enabled=False)
    done = shortcuts.find_replace("red", "blue", apply=True)
    assert done["changed"] == 2
    assert texts() == {"a": ["blue fox", "a blue hat"], "b": ["dark blue door"]}
    assert [s.enabled for s in shortcuts.listing()] == [True, False]


def test_empty_or_blank_find_is_refused(store):
    put("a", "red fox")
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            shortcuts.find_replace(bad, "x", apply=True)
    assert texts() == {"a": ["red fox"]}


def test_regex_characters_are_literal(store):
    put("a", "a.b (x) [y] a+b *", "axb")
    assert shortcuts.find_replace("a.b")["matches"][0]["count"] == 1      # not "any character"
    shortcuts.find_replace(".*", "", apply=True)                          # matches nothing, changes nothing
    shortcuts.find_replace("(x)", r"\1 $1", apply=True)                   # replacement text is literal too
    assert texts()["a"] == ["a.b \\1 $1 [y] a+b *", "axb"]


def test_overlapping_matches_are_taken_left_to_right_once(store):
    put("a", "aaa", "ababab")
    shortcuts.find_replace("aa", "X", apply=True)
    assert texts()["a"] == ["Xa", "ababab"]
    shortcuts.find_replace("aba", "Z", apply=True)
    assert texts()["a"] == ["Xa", "Zbab"]


def test_the_replacement_is_not_searched_again(store):
    put("a", "red", "red red")
    shortcuts.find_replace("red", "red red", apply=True)
    assert texts()["a"] == ["red red", "red red red red"]


def test_a_replacement_that_becomes_empty_is_left_alone_and_reported(store):
    put("a", "red", "red fox")
    done = shortcuts.find_replace("red", "", apply=True)
    assert texts()["a"] == ["red", "fox"]
    assert done["skipped"] == [{"name": "a", "index": 0, "reason": "empty"}]


def test_two_lines_that_become_equal_collapse_to_one(store):
    put("a", "red fox", "blue fox")
    shortcuts.find_replace("red", "blue", apply=True)
    assert texts()["a"] == ["blue fox"]


def test_whitespace_in_the_query_matches_the_single_space_the_library_keeps(store):
    put("a", "red fox")
    assert shortcuts.find_replace("red   fox")["matches"][0]["count"] == 1


def test_a_line_pushed_past_the_length_limit_is_skipped(store):
    put("a", "x" * (shortcuts.MAX_ITEM - 1))
    done = shortcuts.find_replace("x", "xx", apply=True)
    assert done["skipped"][0]["reason"] == "too long" and done["changed"] == 0


def test_unicode_case_insensitive(store):
    put("a", "Éclair and ÉCLAIR")
    assert shortcuts.find_replace("éclair")["matches"][0]["count"] == 2


# ── common phrases ────────────────────────────────────────────────────────

def phrases(**kw):
    return {(r["phrase"], r["shortcuts"], r["uses"]) for r in shortcuts.common_phrases(**kw)}


def test_a_phrase_in_two_shortcuts_is_listed_and_one_in_a_single_shortcut_is_not(store):
    put("a", "soft golden light, a tall man")
    put("b", "SOFT golden light on a hill")
    got = phrases()
    assert ("soft golden light", 2, 2) in got
    assert not any(p[0].lower() in ("a tall", "tall man") for p in got)


def test_the_shorter_phrase_is_dropped_only_when_the_longer_covers_the_same_places(store):
    put("a", "red fox runs")
    put("b", "red fox runs")
    put("c", "red fox sleeps")
    got = phrases()
    assert ("red fox", 3, 3) in got and ("red fox runs", 2, 2) in got
    assert not any(p[0] == "fox runs" for p in got)       # same places as "red fox runs"


def test_punctuation_ends_a_phrase(store):
    put("a", "red fox, big hat")
    put("b", "red fox, big hat")
    assert not any("fox big" in p[0] for p in phrases())
    assert ("red fox", 2, 2) in phrases() and ("big hat", 2, 2) in phrases()


def test_two_copies_in_one_line_are_one_use_and_one_shortcut_alone_is_not_common(store):
    put("a", "red fox red fox red fox")
    assert phrases() == set()
    put("b", "a red fox")
    assert ("red fox", 2, 2) in phrases()


def test_every_listed_phrase_can_be_found_and_the_counts_agree(store):
    put("a", "soft golden light, red fox")
    put("b", "golden light and red fox")
    for r in shortcuts.common_phrases():
        hits = shortcuts.find_replace(r["phrase"])["matches"]
        assert len({m["name"] for m in hits}) == r["shortcuts"] and len(hits) == r["uses"]


def test_nothing_in_an_empty_library(store):
    assert shortcuts.common_phrases() == []
