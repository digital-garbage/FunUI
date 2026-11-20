"""The prompt enhancer: what the model is asked, how its answer is cleaned, and
that every failure hands the ORIGINAL prompt on."""

import json

import pytest

from core import config, shortcuts


@pytest.fixture(autouse=True)
def _library(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SHORTCUTS_FILE", tmp_path / "s.json")
    monkeypatch.setattr(config, "SHORTCUT_CATEGORIES_FILE", tmp_path / "c.json")
    monkeypatch.setattr(config, "ROOT", tmp_path)


def en():
    from modules.conditioning.prompt_enhancer import enhance
    return enhance


class FakeClip:
    """ComfyUI's generate/decode pair, recording what it was asked."""

    def __init__(self, answer="A fox walks.", fail=None):
        self.answer, self.fail, self.seen = answer, fail, {}

    def tokenize(self, text, **kw):
        self.seen = {"text": text, **kw}
        return "tok"

    def generate(self, tokens, **kw):
        if self.fail:
            raise self.fail
        self.seen["gen"] = kw
        return "ids"

    def decode(self, ids, skip_special_tokens=True):
        return self.answer


# --- cleaning ---------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("<think>hmm</think>A fox.", "A fox."),
    ("<think>never ends and eats the answer", ""),
    ("```text\nA fox.\n```", "A fox."),
    ("Here is the prompt: A fox.", "A fox."),
    ('"A fox says \\"hi\\" and runs."', 'A fox says \\"hi\\" and runs.'),
    ('A fox says "hi" and runs.', 'A fox says "hi" and runs.'),
])
def test_clean(raw, want):
    assert en().clean(raw) == want


def test_thinking_is_pulled_out_even_unterminated_or_headless():
    e = en()
    assert e.extract_thinking("<think>a</think>b") == "a"
    assert e.extract_thinking("<think>still going") == "still going"
    assert e.extract_thinking("opened by template</think>answer") == "opened by template"
    assert e.extract_thinking("plain") == ""


def test_a_looping_model_is_cut_at_a_sentence_and_a_long_one_by_the_users_limit():
    e = en()
    loop = "A fox walks across the snowy field. " * 5
    out, cut = e.trim_runaway("Intro sentence here. " + loop)
    assert cut == "repeat"
    assert out.count("A fox walks across the snowy field.") == 2
    long_text = " ".join(f"Sentence number {i} goes on a bit." for i in range(40))
    out, cut = e.trim_runaway(long_text, max_length=16)         # cap 64 chars
    assert cut == "length" and len(out) <= 64 and out.endswith(".")
    assert e.trim_runaway("short", max_length=400) == ("short", False)


# --- chat -------------------------------------------------------------------

def test_chat_shows_only_the_latest_rewrite_and_every_comment():
    e = en()
    rounds = [{"rewrites": {"whole": "v1"}, "comment": "warmer"},
              {"rewrites": {"whole": "v2"}, "comment": "no rain"}]
    block = e.chat_block(rounds)
    assert "v2" in block and "v1" not in block and "- warmer\n- no rain" in block
    assert e.chat_block([{"comment": "  "}, "junk", None]) == ""
    assert e.chat_block([{"rewrites": {"3": "only scene 3"}, "comment": "x"}], scene=3).count("only scene 3") == 1


def test_an_echoed_conversation_is_cut():
    e = en()
    assert e.cut_chat_echo("The fox runs.\n\nUSER FEEDBACK (oldest first):\n- warmer") == "The fox runs."
    assert e.cut_chat_echo("YOUR LATEST REWRITE: The fox runs.") == "The fox runs."


# --- reference --------------------------------------------------------------

def test_reference_uses_what_the_prompt_mentions_else_the_whole_group():
    e = en()
    shortcuts.save({"name": "Rain city", "triggers": ["rc"], "replacements": ["heavy neon rain on asphalt"]})
    shortcuts.save({"name": "Fox", "triggers": ["fx"], "replacements": ["a red fox"]})
    groups = e.sources(["Rain city", "Fox", "Missing"], [], say=lambda m: None)
    assert len(groups) == 1 and len(groups[0]) == 2
    only_rain = e.reference("a walk in the neon street", groups)
    assert "[Shortcut] Rain city" in only_rain and "[Shortcut] Fox" not in only_rain      # a word from its content
    both = e.reference("nothing related", groups)
    assert "Rain city" in both and "Fox" in both                                         # none mentioned: all
    assert e.reference("x", []) == ""


def test_a_missing_shortcut_or_file_is_said_not_silent():
    said = []
    assert en().sources(["Nope"], ["/no/such/file.json"], say=said.append) == []
    assert len(said) == 2 and "Nope" in said[0] and "/no/such/file.json" in said[1]


def test_lorebooks_match_by_keyword_whole_word_and_constants_ride_along(tmp_path):
    e = en()
    book = tmp_path / "lore.json"
    book.write_text(json.dumps({"entries": [
        {"keys": ["sea"], "content": "The sea is cold.", "comment": "Sea"},
        {"keys": ["/ra+in/"], "content": "Rain is soft."},
        {"keys": ["x"], "content": "Always true.", "constant": True}]}))
    groups = e.sources([], [str(book)])
    got = e.reference("a season of rain", groups)          # "sea" is NOT in "season"
    assert "Rain is soft." in got and "The sea is cold." not in got and "Always true." in got


# --- enhance ----------------------------------------------------------------

def test_the_model_gets_instructions_then_the_prompt_and_the_answer_is_returned():
    clip = FakeClip("Here is the prompt: A red fox walks through snow.")
    out, status, info = en().enhance(clip, "a fox", seed=5, max_length=100, do_sample=False)
    assert out == "A red fox walks through snow." and info["ok"]
    assert clip.seen["text"].startswith(en().SYSTEM_PROMPT) and clip.seen["text"].endswith("a fox")
    assert clip.seen["gen"]["seed"] == 5 and clip.seen["gen"]["do_sample"] is False
    assert clip.seen["skip_template"] is False and clip.seen["min_length"] == 1


@pytest.mark.parametrize("clip", [FakeClip(fail=RuntimeError("boom")), FakeClip(""), FakeClip("<think>x</think>"),
                                  None, object()])
def test_every_failure_returns_the_original_prompt(clip):
    out, status, info = en().enhance(clip, "a fox")
    assert out == "a fox" and not info["ok"] and status


@pytest.mark.parametrize("answer", [",,,,,,,,,,,,,,,,,,,,", "...  ---  !!!", "1234 5678", "x ,,,,,,,,,,,,,,,,,,,, ,,,,,"])
def test_an_answer_with_no_words_is_a_failure_not_a_prompt(answer):
    """A text encoder that is not a chat model answered with 400 commas; the run used them as the prompt."""
    out, status, info = en().enhance(FakeClip(answer), "a red fox in the snow")
    assert out == "a red fox in the snow" and not info["ok"] and "no usable words" in status


def test_an_empty_prompt_is_skipped_not_sent():
    clip = FakeClip()
    out, status, _ = en().enhance(clip, "   ")
    assert out == "   " and "empty" in status and clip.seen == {}


class _Tok:
    """ComfyUI's H3 tokenizer shape: takes `images`, swallows everything else."""
    def tokenize_with_weights(self, text, return_word_ids=False, images=[], **kwargs):
        return []


class RealShapeClip(FakeClip):
    """sd.CLIP.generate's real signature: no no_repeat_ngram_size, seed=None default."""
    tokenizer = _Tok()

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95,
                 min_p=0.0, repetition_penalty=1.0, seed=None, presence_penalty=0.0):
        assert seed is not None or not do_sample, "torch.Generator.manual_seed(None) raises"
        self.seen["gen"] = dict(do_sample=do_sample, seed=seed, temperature=temperature)
        return "ids"


def test_the_real_generate_signature_gets_a_seed_and_no_unknown_arguments():
    clip = RealShapeClip("A fox runs.")
    out, status, info = en().enhance(clip, "fox", seed=0)
    assert out == "A fox runs." and info["ok"] and clip.seen["gen"]["seed"]
    assert en().enhance(RealShapeClip("A fox runs."), "fox", seed=7)[2]["ok"]


def test_temperature_zero_is_greedy_not_a_crash():
    clip = RealShapeClip("A fox runs.")
    assert en().enhance(clip, "fox", temperature=0.0)[2]["ok"] and clip.seen["gen"]["do_sample"] is False


def test_a_tokenizer_without_picture_or_template_input_says_so():
    clip = RealShapeClip("A fox runs.")
    _, status, _ = en().enhance(clip, "fox", image=object())
    assert "takes no picture" in status and "no chat template" in status
    assert "image" not in clip.seen and "skip_template" not in clip.seen


def test_chat_revision_goes_in_the_system_turn_and_an_echo_is_cut():
    e = en()
    clip = FakeClip("The fox runs.\nUSER FEEDBACK: warmer")
    chat = e.chat_block([{"rewrites": {"whole": "old"}, "comment": "warmer"}])
    out, _, info = e.enhance(clip, "a fox", chat=chat)
    assert out == "The fox runs." and e.CHAT_SYSTEM in clip.seen["text"]
    assert "ORIGINAL PROMPT:\na fox" in clip.seen["text"] and info["sent"]


# --- style -------------------------------------------------------------------

def test_dynamic_asks_for_motion_and_keeps_every_guard_of_the_default():
    e = en()
    assert e.STYLES["default"] is e.SYSTEM_PROMPT and e.STYLES["dynamic"] is e.DYNAMIC_PROMPT
    d = e.DYNAMIC_PROMPT
    assert "action beat" in d and "change over time" in d and "present progressive" in d
    assert "Never drop, replace or contradict" in d and "Do not invent camera moves or scene cuts unless the user asked" in d
    assert "reproduce it exactly. Never invent speech" in d and "Do not add characters the user did not mention" in d


@pytest.mark.parametrize("style", ["default", "dynamic", "nonsense"])
@pytest.mark.parametrize("typed", ["", "Make it blue."])
@pytest.mark.parametrize("with_chat", [False, True])
@pytest.mark.parametrize("with_reference", [False, True])
def test_style_matrix_typed_instructions_win_then_style_then_default(style, typed, with_chat, with_reference):
    e = en()
    clip = FakeClip("A fox runs.")
    chat = e.chat_block([{"rewrites": {"whole": "old"}, "comment": "faster"}]) if with_chat else ""
    ref = "\n\nREF" if with_reference else ""
    out, _, info = e.enhance(clip, "a fox", system=typed, style=style, chat=chat, reference_text=ref)
    assert out == "A fox runs." and info["ok"]
    sent = clip.seen["text"]
    want = typed or e.STYLES.get(style, e.SYSTEM_PROMPT)           # an unknown style falls back to the default
    assert sent.startswith(want)
    other = e.DYNAMIC_PROMPT if want is not e.DYNAMIC_PROMPT else e.SYSTEM_PROMPT
    assert other not in sent
    assert (e.CHAT_SYSTEM in sent) is with_chat and ("REF" in sent) is with_reference and "a fox" in sent


def test_the_rating_line_closes_the_request_only_when_asked_for(node):
    for on, want in ((True, True), (False, False)):
        clip = FakeClip("A fox sprints.")
        node.FunUIEnhancePrompt.execute(clip, "a fox", enabled=True, seed=3, pass_rating=on, rating_line="RATED liked")
        assert clip.seen["text"].rstrip().endswith("RATED liked") is want


def test_send_everything_into_chat_keeps_earlier_turns_per_conversation(node):
    node.CONVOS.clear()
    run = lambda text, conv, on=True: (lambda c: (node.FunUIEnhancePrompt.execute(c, text, enabled=True, seed=3, to_chat=on, conversation=conv), c.seen["text"])[1])(FakeClip("Answer for " + text + "."))   # noqa: E731
    first = run("fox", "p:0")
    assert "EARLIER IN THIS CONVERSATION" not in first
    second = run("owl", "p:0")
    assert "User: fox\nYou: Answer for fox." in second and "NEW REQUEST:" in second and en().CONVO_SYSTEM in second
    assert "EARLIER IN THIS CONVERSATION" not in run("bat", "p:1")           # Reset: a new conversation
    assert "EARLIER IN THIS CONVERSATION" not in run("cat", "p:0", on=False)  # off: a one-time request
    assert [t["request"] for t in node.CONVOS["p:0"]["turns"]] == ["fox", "owl"]


def test_in_a_conversation_every_comment_goes_in_once_and_stays_in_the_history(node):
    node.CONVOS.clear()
    import json
    def run(text, comments):
        clip = FakeClip("A fox runs.")
        node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="k:0", chat=json.dumps(comments))
        return clip.seen["text"]
    run("fox", [])
    second = run("owl", [{"id": "a", "comment": "warmer", "original": "something else"}])
    assert "Feedback on your earlier rewrite, apply it:\n- warmer" in second and "USER FEEDBACK" not in second
    third = run("bat", [{"id": "a", "comment": "warmer", "original": "something else"}, {"id": "b", "comment": "slower"}])
    assert third.count("- warmer") == 1 and "User: owl\nFeedback on your earlier rewrite, apply it:\n- warmer" in third      # kept in its turn, not repeated
    assert third.rstrip().endswith("- slower") and "User: fox" in third


def test_a_conversation_near_the_context_window_is_summarised_and_the_summary_replaces_the_turns(node):
    node.CONVOS.clear()
    def run(text, **kw):
        clip = FakeClip("A fox runs.")
        node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="c:0", **kw)
        return clip.seen["text"]
    run("fox", context_tokens=4000)
    run("owl", context_tokens=4000)
    assert "SUMMARY OF THE CONVERSATION SO FAR" not in run("bat", context_tokens=4000)      # plenty of room: every turn is sent
    sent = run("cat", context_tokens=80)                                                    # a window this small is full: compact first
    state = node.CONVOS["c:0"]
    assert state["summary"] == "A fox runs." and [t["request"] for t in state["turns"]] == ["cat"]
    assert "SUMMARY OF THE CONVERSATION SO FAR:\nA fox runs." in sent and "User: fox" not in sent
    off = run("dog", context_tokens=80, compact=False)                                      # compacting off: the last turns, no summary
    assert "User: cat" in off
    ctx = node.RUNS[-1]["context"]
    assert ctx["window"] == 80 and ctx["used"] > 0 and ctx["compact"] is False       # shown in the Chat tab: how full the context window is


def test_a_run_reports_the_model_s_own_words_apart_from_the_global_prefix_and_postfix(node):
    node.RUNS.clear()
    node.CONVOS.clear()
    node.FunUIEnhancePrompt.execute(FakeClip("A fox runs."), "PRE a fox POST", enabled=True, seed=3, scope="own", head="PRE", tail="POST", to_chat=True, conversation="x:0", pass_rating=True, rating_line="Rated: Liked")
    run = node.RUNS[-1]
    assert run["after"] == "PRE A fox runs. POST" and run["asked"] == "a fox\nRated: Liked" and run["answered"] == "A fox runs."
    assert list(node.CONVOS["x:0"]["turns"]) == [{"request": "a fox\nRated: Liked", "reply": "A fox runs."}]


class Gemma4_Tokenizer:      # the name ComfyUI's Gemma 4 tokenizers derive from
    prime_empty_thought = True

    def tokenize_with_weights(self, text, return_word_ids=False, image=None, skip_template=True, thinking=False, **kw):
        return text


def test_gemma4_gets_the_conversation_as_real_turns_and_other_encoders_a_flat_block(node):
    node.CONVOS.clear()
    def run(clip, text):
        node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="g:0")
        return clip.seen["text"]
    gemma = FakeClip("A fox runs."); gemma.tokenizer = type("Gemma4SDTokenizer", (Gemma4_Tokenizer,), {})()
    first = run(gemma, "fox")
    assert not first.startswith("<|turn>") and gemma.seen["skip_template"] is False        # nothing earlier yet: the encoder's own template
    second = run(gemma, "owl")
    assert second.startswith("<|turn>system\n") and second.endswith("<|turn>user\nowl<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert "<|turn>user\nfox<turn|>\n<|turn>model\nA fox runs.<turn|>\n<|turn>user\nowl<turn|>" in second and "EARLIER IN THIS CONVERSATION" not in second
    assert gemma.seen["skip_template"] is True
    plain = FakeClip("A fox runs.")
    flat = run(plain, "bat")
    assert "EARLIER IN THIS CONVERSATION" in flat and "<|turn>" not in flat           # no turn format known: the flat block


def test_a_live_chat_reply_uses_its_own_prompt_joins_the_conversation_and_is_shown_as_a_chat_entry(node, monkeypatch):
    node.CONVOS.clear()
    node.RUNS.clear()
    clip = FakeClip("Sure, warmer light it is.")
    monkeypatch.setattr(node, "_own_encoder", lambda f, fam: clip)
    out = node.chat_reply("c:0", "make it warmer?", encoder_file="g.safetensors")
    assert out == {"ok": True, "reply": "Sure, warmer light it is.", "status": out["status"]}
    assert en().CHAT_REPLY_SYSTEM in clip.seen["text"] and en().SYSTEM_PROMPT not in clip.seen["text"] and clip.seen["text"].endswith("make it warmer?")
    run = node.RUNS[-1]
    assert run["kind"] == "chat" and run["asked"] == "make it warmer?" and run["answered"] == out["reply"] and run["conversation"] == "c:0"
    # the next rewrite in that conversation carries the exchange
    rewrite = FakeClip("A fox runs.")
    node.FunUIEnhancePrompt.execute(rewrite, "a fox", enabled=True, seed=3, to_chat=True, conversation="c:0")
    assert "User: make it warmer?\nYou: Sure, warmer light it is." in rewrite.seen["text"]
    # no encoder picked / no conversation: said, nothing runs
    assert not node.chat_reply("c:0", "hi", encoder_file="")["ok"] and not node.chat_reply("", "hi", encoder_file="g")["ok"]


def test_the_model_s_own_context_size_is_read_when_it_reports_one():
    class Cfg: max_position_embeddings = 32768
    class Inner: config = Cfg()
    class Stage: transformer = Inner()
    class Clip: cond_stage_model = Stage()
    assert en().context_window(Clip()) == 32768 and en().context_window(object()) == 0


def test_the_node_has_a_style_input_last_and_passes_it_on(node):
    schema = node.FunUIEnhancePrompt.define_schema()
    assert [i.id for i in schema.inputs][-13:] == ["style", "scope", "last_frame", "to_chat", "conversation", "compact", "context_tokens", "pass_rating", "comment_button", "rating_prefix", "rating_line", "head", "tail"]
    clip = FakeClip("A fox sprints.")
    node.FunUIEnhancePrompt.execute(clip, "a fox", enabled=True, seed=3, style="dynamic")
    assert en().DYNAMIC_PROMPT in clip.seen["text"]
    clip = FakeClip("A fox walks.")
    node.FunUIEnhancePrompt.execute(clip, "a fox", enabled=True, seed=3)
    assert clip.seen["text"].startswith(en().SYSTEM_PROMPT), "a workflow saved before Style rewrites as before"


def test_the_defaults_route_lists_the_styles():
    from modules.conditioning import prompt_enhancer as mod
    table = {}

    class T:
        def get(self, path):
            return lambda fn: table.setdefault(path, fn)

        post = get

    class W:
        @staticmethod
        def json_response(d):
            return d
    mod.routes(T(), "/m", W)
    import asyncio
    got = asyncio.run(table["/m/defaults"](None))
    assert set(got["styles"]) == {"default", "dynamic"} and got["instructions"] == got["styles"]["default"]


# --- the node ---------------------------------------------------------------

@pytest.fixture
def node(comfyui):
    from modules.conditioning.prompt_enhancer import nodes
    nodes.RUNS.clear()
    return nodes


def test_the_node_passes_through_when_off_and_never_touches_the_model(node):
    clip = FakeClip()
    out = node.FunUIEnhancePrompt.execute(clip, "a fox", enabled=False)
    assert tuple(out.args) == ("a fox", "off") and clip.seen == {}
    assert node.RUNS[-1]["after"] == "a fox", "the Enhance tab still shows the final prompt"


def test_the_node_enhances_records_the_run_and_warns_on_failure(node, caplog):
    out = node.FunUIEnhancePrompt.execute(FakeClip("A red fox."), "a fox", enabled=True, seed=3)
    assert out.args[0] == "A red fox." and node.RUNS[-1]["after"] == "A red fox."
    from core import log
    log._reset()
    bad = node.FunUIEnhancePrompt.execute(FakeClip(fail=RuntimeError("x")), "a fox", enabled=True)
    assert bad.args[0] == "a fox" and node.RUNS[-1]["ok"] is False
    assert any("original prompt" in r["message"] for r in log.history())


def test_seed_zero_is_never_cached_a_set_seed_is(node):
    f = node.FunUIEnhancePrompt.fingerprint_inputs
    assert isinstance(f(seed=0, enabled=True), float)
    assert f(seed=7, enabled=True) == "" and f(seed=0, enabled=False) == ""


def test_the_readout_routes_are_provided_and_answer():
    from modules.conditioning import prompt_enhancer as mod
    assert mod.PROVIDES["routes"] is mod.routes
    table = {}

    class T:
        def get(self, path):
            return lambda fn: table.setdefault(path, fn)

        post = get

    class W:
        @staticmethod
        def json_response(d):
            return d
    mod.routes(T(), "/m", W)
    assert set(table) == {"/m/defaults", "/m/runs", "/m/chat", "/m/encoders"}


def test_a_tokenizer_that_reads_the_picture_out_of_kwargs_gets_it():
    class Qwen:                      # Qwen3-VL's shape: names only `images`, reads kwargs.get("image")
        def tokenize_with_weights(self, text, return_word_ids=False, images=[], **kwargs):
            return kwargs.get("image")

    class Clip(RealShapeClip):
        tokenizer = Qwen()
    clip = Clip("A fox runs.")
    _, status, _ = en().enhance(clip, "fox", image=object())
    assert "takes no picture" not in status and "image" in clip.seen


def test_a_picked_encoder_that_will_not_load_keeps_the_prompt_and_says_why(monkeypatch):
    from modules.conditioning.prompt_enhancer import nodes
    def boom(file, family):
        raise RuntimeError("no such family")
    monkeypatch.setattr(nodes, "_own_encoder", boom)
    out = nodes.FunUIEnhancePrompt.execute(clip=object(), text="a fox", enabled=True, encoder_file="x.gguf")
    assert out.args[0] == "a fox" and out.args[1] == "encoder failed"
    assert "x.gguf" in nodes.RUNS[-1]["status"] and not nodes.RUNS[-1]["ok"]


def test_a_gone_encoder_file_is_text_the_graph_does_not_refuse(comfyui):
    from modules.conditioning.prompt_enhancer import nodes
    schema = nodes.FunUIEnhancePrompt.define_schema()
    for name in ("encoder_file", "encoder_family"):
        inp = next(i for i in schema.inputs if i.id == name)
        assert inp.get_io_type() == "STRING"            # a list would refuse a vanished file even with Enhance off


# --- only the text between the global prefix and postfix; markup kept ------------

def _run(node, answer, text, **kw):
    clip = FakeClip(answer)
    out = node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, **kw)
    return out.args, clip


def test_own_scope_sends_only_the_middle_and_puts_prefix_and_postfix_back_as_written(node):
    text = "Cinematic, 35mm. a fox runs. No text on screen."
    (out, _), clip = _run(node, "A red fox sprints through snow.", text, scope="own", head="Cinematic, 35mm.", tail="No text on screen.")
    assert out == "Cinematic, 35mm. A red fox sprints through snow. No text on screen."
    sent = clip.seen["text"]
    assert sent.endswith("a fox runs.") and "Cinematic" not in sent and "No text" not in sent
    assert node.RUNS[-1]["before"] == text and node.RUNS[-1]["after"] == out


def test_whole_scope_is_the_default_and_rewrites_everything(node):
    (out, _), clip = _run(node, "Rewritten all of it.", "Pre. a fox. Post.", head="Pre.", tail="Post.")
    assert out == "Rewritten all of it." and "Pre." in clip.seen["text"]


def test_own_scope_with_no_prefix_or_postfix_is_just_the_text(node):
    (out, _), _ = _run(node, "A red fox.", "a fox", scope="own")
    assert out == "A red fox."


def test_own_scope_that_cannot_find_its_parts_leaves_the_prompt_alone(node):
    clip = FakeClip("never used")
    out = node.FunUIEnhancePrompt.execute(clip, "a fox runs.", enabled=True, scope="own", head="Pre.", tail="")
    assert out.args[0] == "a fox runs." and clip.seen == {} and node.RUNS[-1]["ok"] is False


def test_a_failed_rewrite_in_own_scope_returns_the_whole_original(node):
    (out, _), _ = _run(node, "", "Pre. a fox. Post.", scope="own", head="Pre.", tail="Post.")
    assert out == "Pre. a fox. Post."


@pytest.mark.parametrize("scope", ["whole", "own"])
def test_markup_spans_never_reach_the_model_and_come_back_intact(node, scope):
    text = "Pre. a (fox:1.5) and [cat|dog] run, then ⟦a¦b⟧. Post."
    kw = {"scope": "own", "head": "Pre.", "tail": "Post."} if scope == "own" else {}
    (out, _), clip = _run(node, "A {M1} and {M2} sprint, then {M3}.", text, **kw)
    sent = clip.seen["text"]
    assert "(fox:1.5)" not in sent and "[cat|dog]" not in sent and "⟦" not in sent and "{M1}" in sent
    assert "A (fox:1.5) and [cat|dog] sprint, then ⟦a¦b⟧." in out
    assert out.startswith("Pre.") or scope == "whole"
    assert out.endswith("Post.") or scope == "whole"


def test_a_rewrite_that_loses_or_doubles_a_mark_keeps_the_prompt(node):
    text = "a [cat|dog] runs"
    for answer in ("A cat runs.", "A {M1} {M1} runs.", "A {M2} runs."):
        (out, status), _ = _run(node, answer, text)
        assert out == text and "marks" in status and node.RUNS[-1]["ok"] is False


def test_build_parts_finds_prefix_and_postfix_in_the_expanded_prompt():
    from core import prompt_build
    kw = dict(anchor="Cinematic.", postfix="No text.", variables=[{"name": "who", "text": "a fox"}], seed=1)
    full = prompt_build.build("$who runs [x|y]", **kw)
    parts = prompt_build.build_parts("$who runs [x|y]", **kw)
    assert parts == {"head": "Cinematic.", "tail": "No text."} and full.startswith("Cinematic.") and full.endswith("No text.")
    assert prompt_build.build_parts("hi", anchor="", postfix="") == {"head": "", "tail": ""}
    assert prompt_build.build_parts("hi", anchor="A.", postfix="P.", postfix_enabled=False) == {"head": "A.", "tail": ""}


def test_the_apps_own_shortcuts_file_is_a_valid_reference_file(tmp_path):
    shortcuts.save({"name": "Rain city", "triggers": ["rc"], "replacements": ["heavy neon rain on asphalt"]})
    saved = config.SHORTCUTS_FILE
    assert isinstance(json.loads(saved.read_text()), list)          # the shape the app itself writes
    said = []
    groups = en().sources([], [str(saved)], say=said.append)
    assert said == [] and len(groups) == 1 and "Rain city" in groups[0][0]["block"]
    scalar = tmp_path / "x.json"
    scalar.write_text("3")
    assert en().sources([], [str(scalar)], say=said.append) == [] and "not a lorebook" in said[0]


def test_each_scene_of_a_project_is_enhanced_with_its_own_prefix_split_and_nothing_leaks(node):
    """Made-up scenes through the real expander and the node, one run each, as a multi-scene Generate does."""
    from core import prompt_build
    project = dict(anchor="Film grain.", postfix="No subtitles.", variables=[{"name": "hero", "text": "a fox"}], seed=1)
    scenes = ["$hero runs [left|right] along a river", "the camera rises over $hero", "an empty field"]
    answers = ["A fox sprints {M1} along a river.", "The camera climbs over the fox.", "A wide empty field at dawn."]
    outs = []
    for text, answer in zip(scenes, answers):
        full = prompt_build.build(text, **project)
        parts = prompt_build.build_parts(text, **project)
        clip = FakeClip(answer)
        out = node.FunUIEnhancePrompt.execute(clip, full, enabled=True, seed=3, scope="own", **parts).args[0]
        sent = clip.seen["text"]
        assert "Film grain." not in sent and "No subtitles." not in sent
        assert out.startswith("Film grain. ") and out.endswith(" No subtitles.") and out.count("Film grain.") == 1
        outs.append(out)
    assert "⟦left¦right⟧" in outs[0]                                        # the alternation came back as the expander wrote it
    assert "river" not in outs[1] + outs[2] and "fox sprints" not in outs[1] + outs[2]
    assert [o.count("No subtitles.") for o in outs] == [1, 1, 1]

