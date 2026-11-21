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
    assert list(node.CONVOS["x:0"]["turns"]) == [{"request": "a fox\nRated: Liked", "reply": "A fox runs.", "prompt": "a fox", "feedback": [], "rating": "Rated: Liked"}]


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
    assert out == {"ok": True, "reply": "Sure, warmer light it is.", "status": out["status"], "remember": [], "forget": [], "compacted": False}
    assert en().CHAT_REPLY_SYSTEM in clip.seen["text"] and en().SYSTEM_PROMPT not in clip.seen["text"] and clip.seen["text"].endswith("make it warmer?")
    run = node.RUNS[-1]
    assert run["kind"] == "chat" and run["asked"] == "make it warmer?" and run["answered"] == out["reply"] and run["conversation"] == "c:0"
    # the next rewrite in that conversation carries the exchange
    rewrite = FakeClip("A fox runs.")
    node.FunUIEnhancePrompt.execute(rewrite, "a fox", enabled=True, seed=3, to_chat=True, conversation="c:0")
    assert "User: make it warmer?\nYou: Sure, warmer light it is." in rewrite.seen["text"]
    # no encoder picked / no conversation: said, nothing runs
    assert not node.chat_reply("c:0", "hi", encoder_file="")["ok"] and not node.chat_reply("", "hi", encoder_file="g")["ok"]


def test_the_model_edits_the_standing_notes_from_the_chat_and_every_rewrite_carries_them(node, monkeypatch):
    node.CONVOS.clear()
    clip = FakeClip("Done, warmer from now on.\nREMEMBER: warm light\nFORGET: 1")
    monkeypatch.setattr(node, "_own_encoder", lambda f, fam: clip)
    out = node.chat_reply("n:0", "always use warm light", encoder_file="g.safetensors", notes="no rain\n\nwide shots")
    assert out["reply"] == "Done, warmer from now on." and out["remember"] == ["warm light"] and out["forget"] == ["1"]
    assert "1. no rain\n2. wide shots" in clip.seen["text"] and "REMEMBER:" in clip.seen["text"]      # numbered, so a note can be dropped by number
    rewrite = FakeClip("A fox runs.")
    node.FunUIEnhancePrompt.execute(rewrite, "a fox", enabled=True, seed=3, notes="warm light\nwide shots")
    assert "STANDING NOTES from the user (they always apply):\n- warm light\n- wide shots" in rewrite.seen["text"]
    plain = FakeClip("A fox runs.")
    node.FunUIEnhancePrompt.execute(plain, "a fox", enabled=True, seed=3)
    assert "STANDING NOTES" not in plain.seen["text"]


def test_summarising_works_to_a_practical_window_even_when_the_model_reports_a_huge_one(node):
    class Cfg: max_position_embeddings = 131072
    class Stage: config = Cfg()
    class Big(FakeClip):
        cond_stage_model = Stage()
    node.CONVOS.clear()
    node.RUNS.clear()
    clip = Big("A fox runs.")
    node.FunUIEnhancePrompt.execute(clip, "fox", enabled=True, seed=3, to_chat=True, conversation="w:0")
    assert node.RUNS[-1]["context"]["window"] == 32768 and node.RUNS[-1]["compacted"] is False       # the model's 131072 is a ceiling, not the budget
    node.FunUIEnhancePrompt.execute(clip, "owl", enabled=True, seed=3, to_chat=True, conversation="w:0", context_tokens=120)
    assert node.RUNS[-1]["compacted"] is True and "summarised" in node.RUNS[-1]["status"] and node.RUNS[-1]["summary"] == "A fox runs."      # the summary itself is shown in the Chat tab             # said in the status and flagged for the app


def test_the_model_s_own_context_size_is_read_when_it_reports_one():
    class Cfg: max_position_embeddings = 32768
    class Inner: config = Cfg()
    class Stage: transformer = Inner()
    class Clip: cond_stage_model = Stage()
    assert en().context_window(Clip()) == 32768 and en().context_window(object()) == 0


def test_the_node_has_a_style_input_last_and_passes_it_on(node):
    schema = node.FunUIEnhancePrompt.define_schema()
    assert [i.id for i in schema.inputs][-15:] == ["style", "scope", "last_frame", "to_chat", "conversation", "scene", "compact", "context_tokens", "pass_rating", "comment_button", "notes", "rating_prefix", "rating_line", "head", "tail"]
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
    assert set(table) == {"/m/defaults", "/m/runs", "/m/chat", "/m/chat_stream", "/m/engine", "/m/engine/load", "/m/engine/download", "/m/engine/unload", "/m/engine/files", "/m/engine/live", "/m/engine/models", "/m/engine/delete", "/m/encoders", "/m/edit_decision"}


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
    assert sent.endswith("a fox runs.") and sent.count("Cinematic") == 1 and "Fixed text before it: Cinematic, 35mm." in sent       # told once, in the instructions, never inside the text it rewrites
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
        assert sent.count("Film grain.") == 1 and sent.count("No subtitles.") == 1 and "Fixed text before it: Film grain." in sent
        assert out.startswith("Film grain. ") and out.endswith(" No subtitles.") and out.count("Film grain.") == 1
        outs.append(out)
    assert "⟦left¦right⟧" in outs[0]                                        # the alternation came back as the expander wrote it
    assert "river" not in outs[1] + outs[2] and "fox sprints" not in outs[1] + outs[2]
    assert [o.count("No subtitles.") for o in outs] == [1, 1, 1]



def test_notes_lines_survive_the_markdown_a_small_model_adds():
    body, add, drop = en().parse_notes("Sure, warmer.\n**REMEMBER:** warm light\n- Remember : \"no rain\"\n> FORGET: 2\n(forget: 3)")
    assert body == "Sure, warmer." and add == ["warm light", "no rain"] and drop == ["2", "3"]
    assert en().parse_notes("I will remember: that is fine.")[1:] == ([], [])      # mid-sentence stays an answer


def test_a_chat_reply_keeps_the_models_thinking_for_the_chat_tab(node, monkeypatch):
    node.CONVOS.clear()
    node.RUNS.clear()
    monkeypatch.setattr(node, "_own_encoder", lambda f, fam: FakeClip("<think>warm means orange</think>Sure, warmer."))
    out = node.chat_reply("c:1", "warmer?", encoder_file="g.safetensors")
    assert out["reply"] == "Sure, warmer." and node.RUNS[-1]["thinking"] == "warm means orange"


def test_a_new_request_for_the_same_scene_replaces_its_earlier_turn(node):
    node.CONVOS.clear()
    run = lambda clip, text, scene: node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="p:0", scene=scene)
    run(FakeClip("A red fox runs."), "a fox", "s1")
    run(FakeClip("A dog sleeps."), "a dog", "s2")
    edited = FakeClip("An owl glides.")
    run(edited, "an owl", "s1")
    seen = edited.seen["text"]
    assert "a fox" not in seen and "A red fox runs." not in seen         # the abandoned version of scene 1 is gone
    assert "User: a dog\nYou: A dog sleeps." in seen                       # another scene stays
    assert [t["request"] for t in node.CONVOS["p:0"]["turns"]] == ["a dog", "an owl"]
    run(FakeClip(""), "a cat", "s2")                                        # a failed run replaces nothing
    assert [t["request"] for t in node.CONVOS["p:0"]["turns"]] == ["a dog", "an owl"]


def test_the_note_taker_sees_comments_ratings_and_answers_labelled_and_is_not_penalised_for_repeating_words(node):
    node.CONVOS.clear()
    run = lambda clip, text, **kw: node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="c:2", **kw)
    run(FakeClip("A red fox runs."), "a fox", rating_line="The previous scene was rated: Liked", pass_rating=True)
    node.CONVOS["c:2"]["seen"].clear()
    notes = FakeClip("SCENES: a red fox.")
    seen = {}
    real = en().generate
    def spy(clip, system, user, **kw):
        seen.update(system=system, user=user, kw=kw)
        return real(clip, system, user, **kw)
    import unittest.mock as mock
    with mock.patch.object(en(), "generate", spy):
        assert en().compact(notes, "", list(node.CONVOS["c:2"]["turns"])) == "SCENES: a red fox."
    assert "Person's request: a fox" in seen["user"] and "Rating of the scene before: The previous scene was rated: Liked" in seen["user"] and "Writer's answer: A red fox runs." in seen["user"]
    assert seen["user"].endswith(en().COMPACT_ASK) and "PERSON'S FEEDBACK" in seen["system"]
    assert seen["kw"]["repetition_penalty"] == 1.0 and seen["kw"]["do_sample"] is False


def test_gemma4_writes_the_notes_thinking_with_its_own_sampling_and_other_encoders_plainly():
    gemma = FakeClip("<think>x</think>SCENES: a fox."); gemma.tokenizer = type("Gemma4SDTokenizer", (Gemma4_Tokenizer,), {})()
    turn = {"request": "a fox", "reply": "A red fox."}
    assert en().compact(gemma, "", [turn]) == "SCENES: a fox."
    assert gemma.seen["thinking"] is True
    assert {k: gemma.seen["gen"][k] for k in ("do_sample", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "max_length")} == {
        "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05, "repetition_penalty": 1.0, "presence_penalty": 0.0, "max_length": en().COMPACT_THINK_TOKENS}
    plain = FakeClip("SCENES: a fox.")
    en().compact(plain, "", [turn])
    assert plain.seen["gen"]["do_sample"] is False and plain.seen["gen"]["repetition_penalty"] == 1.0 and plain.seen["gen"]["max_length"] == en().COMPACT_TOKENS


def test_every_run_gets_a_chat_id_that_only_rises(node):
    node.RUNS.clear()
    for _ in range(3):
        node.FunUIEnhancePrompt.execute(FakeClip("A fox."), "a fox", enabled=True, seed=3)
    ids = [r["chat_id"] for r in node.RUNS]
    assert ids == sorted(set(ids)) and len(ids) == 3


def test_every_system_prompt_can_be_replaced_and_empty_means_the_built_in_one(node, monkeypatch):
    node.CONVOS.clear()
    assert set(en().PROMPT_FIELDS) == {"chat_prompt", "notes_prompt", "compact_prompt", "revision_prompt", "conversation_prompt", "turns_prompt", "marks_prompt", "fixed_prompt", "edit_prompt", "refine_prompt"}
    assert all(en().defaults()[k].strip() for k in en().PROMPT_FIELDS) and en().pick({"chat_prompt": "  "}, "chat_prompt") == en().CHAT_REPLY_SYSTEM
    run = lambda clip, text, **kw: node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="s:0", **kw)
    # conversation rules, flat and as turns; the markup rule
    run(FakeClip("A fox."), "a fox")
    flat = FakeClip("An owl.")
    run(flat, "an (owl:1.5)", conversation_prompt="MY FLAT RULES", marks_prompt="MY MARKS")
    assert "MY FLAT RULES" in flat.seen["text"] and en().CONVO_SYSTEM not in flat.seen["text"]
    assert "MY MARKS" in flat.seen["text"] and en().MARKS_RULE not in flat.seen["text"]
    gemma = FakeClip("A bat."); gemma.tokenizer = type("Gemma4SDTokenizer", (Gemma4_Tokenizer,), {})()
    run(gemma, "a bat", turns_prompt="MY TURN RULES")
    assert "MY TURN RULES" in gemma.seen["text"] and en().CONVO_NATIVE_SYSTEM not in gemma.seen["text"]
    # the revision rules for comments outside a conversation
    rev = FakeClip("A fox.")
    node.FunUIEnhancePrompt.execute(rev, "a fox", enabled=True, seed=3, chat=json.dumps([{"comment": "warmer", "rewrites": {"whole": "A fox."}}]), revision_prompt="MY REVISION")
    assert "MY REVISION" in rev.seen["text"] and en().CHAT_SYSTEM not in rev.seen["text"]
    # the chat answer and the notes rule, and the note-taker
    chat = FakeClip("Sure.")
    monkeypatch.setattr(node, "_own_encoder", lambda f, fam: chat)
    node.chat_reply("s:1", "hi", encoder_file="g", prompts={"chat_prompt": "MY CHAT", "notes_prompt": "MY NOTES"})
    assert "MY CHAT" in chat.seen["text"] and "MY NOTES" in chat.seen["text"] and en().CHAT_REPLY_SYSTEM not in chat.seen["text"]
    keep = FakeClip("SCENES: x.")
    en().compact(keep, "", [{"request": "a", "reply": "b"}], prompts={"compact_prompt": "MY NOTE-TAKER"})
    assert "MY NOTE-TAKER" in keep.seen["text"] and en().COMPACT_SYSTEM not in keep.seen["text"]


def test_macros_in_a_system_prompt_become_what_the_conversation_holds(node, monkeypatch):
    node.CONVOS.clear()
    assert set(en().macro_values()) == set(en().MACROS) and en().expand_macros("[[NOTES]] [[NOPE]]", en().macro_values()) == "(nothing yet: none was given so far) [[NOPE]]"
    run = lambda clip, text, **kw: node.FunUIEnhancePrompt.execute(clip, text, enabled=True, seed=3, to_chat=True, conversation="m:0", **kw)
    run(FakeClip("A red fox."), "a fox")
    run(FakeClip("A grey owl."), "an owl")
    comments = json.dumps([{"comment": "warmer"}, {"comment": "no rain"}])
    clip = FakeClip("A bat.")
    run(clip, "a bat", instructions="PREV=[[PREVIOUS_MODEL_OUTPUT]] HIST=[[MODEL_OUTPUTS_HISTORY]] NOW=[[CURRENT_PROMPT]] LASTC=[[LAST_USER_COMMENT]] ALLC=[[COMMENTS_HISTORY]] PP=[[PREVIOUS_PROMPT]] N=[[NOTES]]",
        chat=comments, notes="warm light")
    text = clip.seen["text"]
    assert "PREV=A grey owl. HIST=- A red fox. NOW=a bat LASTC=no rain ALLC=- warmer PP=an owl N=1. warm light" in text
    # a chat answer sees the message it answers as the last one, and the comments the app sends
    chat = FakeClip("Sure.")
    monkeypatch.setattr(node, "_own_encoder", lambda f, fam: chat)
    node.chat_reply("m:0", "first?", encoder_file="g", comments=["warmer"], prompts={"chat_prompt": "LAST=[[LAST_CHAT_MESSAGE]] OUT=[[PREVIOUS_MODEL_OUTPUT]] C=[[LAST_USER_COMMENT]]"})
    node.chat_reply("m:0", "second?", encoder_file="g", prompts={"chat_prompt": "LAST=[[LAST_CHAT_MESSAGE]] HIST=[[CHAT_MESSAGE_HISTORY]] R=[[LAST_CHAT_REPLY]]"})
    assert "LAST=second? HIST=- first? R=Sure." in chat.seen["text"]
    # ratings passed with earlier requests
    node.CONVOS.clear()
    run(FakeClip("A fox runs."), "a", pass_rating=True, rating_line="Rated: Liked")
    run(FakeClip("An owl glides."), "b", pass_rating=True, rating_line="Rated: Disliked")
    rate = FakeClip("A bat flies.")
    run(rate, "c", pass_rating=True, rating_line="Rated: Loved", instructions="NOW=[[RATING]] LAST=[[LAST_RATING]] ALL=[[RATINGS_HISTORY]]")
    assert "NOW=Rated: Loved LAST=Rated: Disliked ALL=- Rated: Liked" in rate.seen["text"]


def test_every_switch_that_is_on_reaches_the_model(comfyui, monkeypatch):
    """Notes, comments, rating, an earlier rewrite and every macro a prompt names are in the request the model is given."""
    from modules.conditioning.prompt_enhancer import nodes

    class Engineish:
        is_engine = native_turns = True
        asked = []

        def __init__(self, *a, **k):
            pass

        def run(self, system, user, rounds=(), **k):
            Engineish.asked.append((system, user, rounds))
            return "A fox runs through snow.", "ok"
    monkeypatch.setattr(nodes, "EngineClip", Engineish)
    monkeypatch.setattr(nodes.ENGINE, "load", lambda s, p="auto": None)
    nodes.CONVOS.clear()
    rules = "Last output: [[PREVIOUS_MODEL_OUTPUT]] Last comment: [[LAST_USER_COMMENT]] Rating: [[RATING]] Notes: [[NOTES]]"
    run = lambda p, scene, comment="": nodes.FunUIEnhancePrompt.execute(            # noqa: E731
        None, p, enabled=True, engine_model="m", conversation="c:1", to_chat=True, pass_rating=True, rating_line="Rating: 4/5 stars",
        chat=json.dumps([{"id": "1", "comment": comment}]) if comment else "", notes="keep it short", scene=scene, instructions=rules)
    run("a fox", "s0")
    first = Engineish.asked[-1][0]
    assert "keep it short" in first and "nothing yet" in first and "Rating: 4/5 stars" in first and "Rating: 4/5 stars" in Engineish.asked[-1][1]
    run("a fox in rain", "s1", "more rain please")
    system, user, rounds = Engineish.asked[-1]
    assert "A fox runs through snow." in system and "more rain please" in system and "keep it short" in system and "Rating: 4/5 stars" in system
    assert "more rain please" in user and "Rating: 4/5 stars" in user and rounds and rounds[0][1] == "A fox runs through snow."
    assert "a fox in rain" in nodes.RUNS[-1]["sent"] and "more rain please" in nodes.RUNS[-1]["sent"] and "EARLIER TURN 1" in nodes.RUNS[-1]["sent"]


def test_in_own_text_mode_the_model_is_told_what_the_fixed_prefix_and_postfix_say(comfyui, monkeypatch):
    from modules.conditioning.prompt_enhancer import nodes
    seen = []

    class Engineish:
        is_engine = native_turns = True

        def __init__(self, *a, **k):
            pass

        def run(self, system, user, rounds=(), **k):
            seen.append((system, user))
            return "a red fox runs", "ok"
    monkeypatch.setattr(nodes, "EngineClip", Engineish)
    monkeypatch.setattr(nodes.ENGINE, "load", lambda s, p="auto": None)
    out = nodes.FunUIEnhancePrompt.execute(None, "cinematic, a fox, 4k", enabled=True, engine_model="m", scope="own", head="cinematic,", tail=", 4k",
                                           instructions="Prefix is [[PROMPT_PREFIX]] and postfix is [[PROMPT_POSTFIX]].")
    system, user = seen[-1]
    assert "Fixed text before it: cinematic," in system and "Fixed text after it: , 4k" in system and "Prefix is cinematic, and postfix is , 4k." in system
    assert user.strip() == "a fox" and out.args[0] == "cinematic, a red fox runs , 4k".replace(" ,", ",") or "a red fox runs" in out.args[0]
    nodes.FunUIEnhancePrompt.execute(None, "a fox", enabled=True, engine_model="m")
    assert "Fixed text" not in seen[-1][0]                                   # no prefix or postfix: nothing to tell


def test_a_note_line_glued_onto_the_end_of_a_sentence_still_counts():
    text, add, drop = en().parse_notes("I will keep it short. REMEMBER: Answer in one short sentence.")
    assert add == ["Answer in one short sentence."] and text == "I will keep it short."
    assert en().parse_notes("Fine.\nFORGET: 2")[2] == ["2"]
    assert en().parse_notes("He said REMEMBER: this is not a note line")[1] == []              # only after a sentence end or on a line of its own


class ScriptedClip(FakeClip):
    """Answers from a list, one per call, and keeps what each call was asked."""

    def __init__(self, answers):
        super().__init__()
        self.answers, self.asked = list(answers), []

    def generate(self, tokens, **kw):
        self.asked.append(self.seen.get("text", ""))
        return [1]

    def decode(self, ids, **kw):
        return self.answers.pop(0) if self.answers else ""


def test_refine_passes_let_the_model_check_and_improve_its_own_rewrite():
    clip = ScriptedClip(["A fox runs.", "The rewrite lost the snow.\nFINAL:\nA red fox runs through deep snow."])
    out, status, info = en().enhance(clip, "a fox in snow", refine=1, seed=1)
    assert out == "A red fox runs through deep snow." and "refined 1x" in status
    assert "YOUR REWRITE:\nA fox runs." in clip.asked[-1] and "WHAT YOU WERE GIVEN:\na fox in snow" in clip.asked[-1]
    none = ScriptedClip(["A fox runs.", ""])                              # a pass that gives nothing leaves the draft as it was
    assert en().enhance(none, "a fox in snow", refine=2, seed=1)[0] == "A fox runs."
    off = ScriptedClip(["A fox runs."])
    assert en().enhance(off, "a fox in snow", seed=1)[0] == "A fox runs." and len(off.asked) == 1       # no passes by default: one call


def test_the_comfy_switch_keeps_the_engine_out_of_a_rewrite_and_a_chat(monkeypatch):
    from .. import nodes
    loaded = []
    monkeypatch.setattr(nodes.ENGINE, "load", lambda s, p="auto": loaded.append(s))
    out = nodes.FunUIEnhancePrompt.execute(None, "a fox", enabled=True, engine_model="a/b", use_comfy=True)
    assert loaded == [] and out.args[0] == "a fox"                                  # no clip to write with here: the prompt passes, the engine was never asked
    assert nodes.chat_reply("c", "hi", engine_model="a/b", use_comfy=True)["ok"] is False and loaded == []


def test_on_the_comfy_path_every_switch_still_reaches_the_encoder_and_the_engine_stays_out(node, monkeypatch):
    """"Use ComfyUI's text generation instead" with every switch on: notes, comment, rating, the earlier rewrite, the fixed prefix and
    postfix, and the conversation as real Gemma turns are in the text the encoder is given."""
    node.CONVOS.clear()
    asked = []
    monkeypatch.setattr(node.ENGINE, "load", lambda *a, **k: asked.append(a))
    gemma = FakeClip("A fox runs through snow.")
    gemma.tokenizer = type("Gemma4SDTokenizer", (Gemma4_Tokenizer,), {})()
    rules = "Last output: [[PREVIOUS_MODEL_OUTPUT]] Last comment: [[LAST_USER_COMMENT]] Rating: [[RATING]] Notes: [[NOTES]]"
    comment = json.dumps([{"id": "1", "comment": "more rain please"}])
    run = lambda text, scene, chat="": node.FunUIEnhancePrompt.execute(            # noqa: E731
        gemma, text, enabled=True, seed=3, engine_model="a/b", use_comfy=True, to_chat=True, conversation="u:0", pass_rating=True, rating_line="Rating: 4/5 stars",
        scope="own", head="PRE", tail="POST", notes="keep it short", scene=scene, chat=chat, instructions=rules, thinking=True)
    run("PRE a fox POST", "s0")
    first = gemma.seen["text"]
    assert "a fox" in first and "keep it short" in first and "Rating: 4/5 stars" in first and "PRE" in first and "POST" in first
    run("PRE a fox in rain POST", "s1", comment)
    second = gemma.seen["text"]
    assert second.startswith("<|turn>system\n") and "<|turn>user\n" in second           # real Gemma turns, the instructions in the system turn
    for needed in ("a fox in rain", "more rain please", "keep it short", "Rating: 4/5 stars", "A fox runs through snow.", "PRE", "POST", "<|turn>model\nA fox runs through snow.<turn|>"):
        assert needed in second, needed
    assert gemma.seen["thinking"] is True and asked == []
