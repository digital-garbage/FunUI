"""Prompt markup, with H3's REAL tokenizer (offline) and the tiny real H3 model.

The encoder is a stand-in -- one fixed vector per token id -- because Qwen3-VL-32B
is not a unit-test dependency; placement depends on the tokenizer and the tags,
which are real here.
"""

import math

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class FakeH3Clip:
    def __init__(self):
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        self.tokenizer = MiniMaxH3Tokenizer(embedding_directory=None, tokenizer_data={})

    def tokenize(self, text):
        return self.tokenizer.tokenize_with_weights(text)

    def encode_from_tokens_scheduled(self, tokens):
        ids = [t[0] for t in tokens["qwen3vl_32b"][0]]
        rows = torch.stack([torch.randn(48, generator=torch.Generator().manual_seed(int(i)))
                            for i in ids]).unsqueeze(0)
        return [[rows, {"minimax_token_tags": torch.ones(len(ids), dtype=torch.long)}]]


def _encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def _apply(tiny, raw):
    from modules.conditioning.prompt_markup.nodes import (FunUIApplyPromptMarkup,
                                                          FunUIPromptMarkup)
    clean, markup, _ = FunUIPromptMarkup.execute(raw).result
    clip = FakeH3Clip()
    positive = _encode(clip, clean)
    tiny.context = positive[0][0]
    model, pos, status = FunUIApplyPromptMarkup.execute(
        tiny.patcher, clip, positive, {"samples": tiny.video}, markup).result
    return model, pos, status, positive


def _spy_masks(tiny, model):
    """Every (q rows, mask) the backend receives, via an override UNDER ours."""
    from core import dit_hooks
    seen = []

    def spy(func, q, k, v, *a, mask=None, **kw):
        seen.append((q.shape[2], mask))
        return func(q, k, v, *a, mask=mask, **kw)

    under = tiny.patcher.clone()
    dit_hooks.add_attention_override(under, "spy", spy)
    ours = model.model_options["transformer_options"]["optimized_attention_override"]
    # Put ours on top of the spy.
    under.model_options["transformer_options"]["optimized_attention_override"] = (
        lambda func, *a, **kw: ours(lambda *x, **y: under_spy(func, *x, **y), *a, **kw))
    under_spy = lambda func, *a, **kw: spy(func, *a, **kw)      # noqa: E731
    tiny.run(under)
    return seen


def test_the_node_strips_markup_and_reports_it():
    from modules.conditioning.prompt_markup.nodes import FunUIPromptMarkup
    clean, markup, status = FunUIPromptMarkup.execute("a (red:1.5) car [sits|runs]").result
    assert clean == "a red car sits while runs"
    assert markup["variants"] == ["a red car sits", "a red car runs"]
    assert status == "1 weighted, 2 alternative"


def test_a_weight_biases_exactly_that_words_keys(tiny_h3):
    model, _p, status, _pos = _apply(tiny_h3, "a (red:1.5) car drives")
    assert status == "1 weight(s)"
    masks = [m for n, m in _spy_masks(tiny_h3, model) if m is not None]
    assert masks, "the bias never reached attention"
    row = masks[0].reshape(-1)
    # tokens: "a" " red" " car" " drives" -> "red" is token 1, prompt at row 0
    assert row[1].item() == pytest.approx(math.log(1.5), rel=1e-4)
    assert torch.count_nonzero(row).item() == 1


def test_it_changes_the_output_and_weight_one_does_not(tiny_h3):
    base_model, _p, _s, pos = _apply(tiny_h3, "a red car drives")
    base = tiny_h3.run(base_model)
    weighted, *_ = _apply(tiny_h3, "a (red:3) car drives")
    assert not torch.allclose(tiny_h3.run(weighted)[0], base[0])


def test_the_options_of_a_pipe_become_one_prompt_each_and_the_main_prompt_holds_both(tiny_h3):
    from modules.conditioning.prompt_markup.nodes import SIMUL
    model, _p, status, pos = _apply(tiny_h3, "a girl [dances|zooms in] at dusk")
    assert status == "2 simultaneous option(s)"
    got = model.model_options["transformer_options"][SIMUL]
    assert got["cond_len"] == int(pos[0][0].shape[1]) and len(got["options"]) == 2
    # Each option is a whole conditioning of the encoded size; the two differ only in the prompt rows.
    assert all(o.shape == pos[0][0].shape for o in got["options"])
    differ = (got["options"][0] != got["options"][1]).any(dim=-1)[0].nonzero().flatten()
    assert len(differ) and int(differ.max()) - int(differ.min()) < pos[0][0].shape[1]
    assert not torch.allclose(got["options"][0], got["options"][1])
    plain, *_ = _apply(tiny_h3, "a girl dances at dusk")
    assert plain.model_options.get("transformer_options", {}).get(SIMUL) is None


def test_the_parser_picks_options_and_keeps_weights_aligned():
    from modules.conditioning.prompt_markup import markup as mk
    got = mk.parse("a (red:1.5) car [sits|runs] now [x:2] [one|two|three]")
    assert got["clean"] == "a red car sits while runs now x one while two while three"
    assert got["variants"] == ["a red car sits now x one", "a red car runs now x two"]
    assert got["extra"] == 1
    assert [(got["clean"][a:b], w) for a, b, w in got["weighted"]] == [("red", 1.5), ("x", 2.0)]
    assert mk.parse("a [only|] one")["variants"] == []          # nothing to be simultaneous with
    assert mk.parse(r"a \[sits|runs]")["variants"] == []        # an escaped bracket is text


def test_an_encoder_no_model_module_knows_is_refused_out_loud(tiny_h3):
    from modules.conditioning.prompt_markup.nodes import (FunUIApplyPromptMarkup,
                                                          FunUIPromptMarkup)
    _clean, markup, _ = FunUIPromptMarkup.execute("a (red:2) car").result

    class OtherClip:
        tokenizer = object()

    positive = [[torch.zeros(1, 4, 48), {}]]
    model, pos, status = FunUIApplyPromptMarkup.execute(
        tiny_h3.patcher, OtherClip(), positive, {"samples": tiny_h3.video}, markup).result
    assert status.startswith("not applied") and pos is positive


def test_the_phrases_it_placed_are_published_for_anything_that_wants_to_look_at_them(tiny_h3):
    from modules.conditioning.prompt_markup.nodes import PHRASES
    model, _p, _s, pos = _apply(tiny_h3, "a (red:1.5) car (drives:2)")
    published = model.model_options["transformer_options"][PHRASES]
    assert published["cond_len"] == int(pos[0][0].shape[1])
    assert [b - a for a, b in published["spans"]] == [1, 1]          # "red" and "drives": one token each
    plain, *_ = _apply(tiny_h3, "a red car drives")
    assert PHRASES not in plain.model_options.get("transformer_options", {})


def test_the_main_prompt_can_read_both_options_joined_by_while_or_a_space_or_just_one():
    from modules.conditioning.prompt_markup import markup as mk
    text = "a girl [dances|zooms in] at dusk"
    assert mk.parse(text)["clean"] == "a girl dances while zooms in at dusk"
    assert mk.parse(text, main="space")["clean"] == "a girl dances zooms in at dusk"
    assert mk.parse(text, main="first")["clean"] == "a girl dances at dusk"
    assert mk.parse(text, main="second")["clean"] == "a girl zooms in at dusk"
    for main in ("while", "space", "first", "second"):                       # the options themselves never change
        assert mk.parse(text, main=main)["variants"] == ["a girl dances at dusk", "a girl zooms in at dusk"]


def test_the_node_reads_the_choice_from_step_merging_only_while_it_is_switched_on(comfyui_registry=None):
    from modules.conditioning.prompt_markup.nodes import FunUIPromptMarkup
    text = "a girl [dances|zooms in] at dusk"
    run = lambda nm: FunUIPromptMarkup.execute(text, {"noise_merge": nm}).result[0]          # noqa: E731
    assert run({"enabled": True, "main": "space"}) == "a girl dances zooms in at dusk"
    assert run({"enabled": True, "main": "second"}) == "a girl zooms in at dusk"
    assert run({"enabled": True, "actions": False, "main": "second"}) == "a girl dances while zooms in at dusk"
    assert run({"enabled": False, "main": "second"}) == "a girl dances while zooms in at dusk"
    off = FunUIPromptMarkup.execute(text, {"_off": {"all": True}, "noise_merge": {"enabled": True, "main": "second"}}).result[0]
    assert off == "a girl dances while zooms in at dusk"
