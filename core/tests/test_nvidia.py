from core import nvidia


def test_a_loaded_effect_is_kept_between_jobs_and_replaced_by_a_different_setting(monkeypatch):
    monkeypatch.setattr(nvidia, "_held", {})
    built = []
    make = lambda tag: (lambda: built.append(tag) or tag)
    assert nvidia.held("k", (1, 2), make("a")) == "a" and nvidia.held("k", (1, 2), make("b")) == "a"
    assert nvidia.held("k", (1, 3), make("c")) == "c" and built == ["a", "c"]       # one per kind: the old one is let go
    assert nvidia.held("other", (1, 3), make("d")) == "d" and nvidia.held("k", (1, 3), make("e")) == "c"
