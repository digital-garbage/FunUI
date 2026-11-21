"""The enhancer's own transformers engine: how it reads any model's thinking and tool calls, and that a model streams through it."""

import pytest

pytest.importorskip("transformers")

from modules.conditioning.prompt_enhancer import engine as eng      # noqa: E402


def pieces(dialect, chunks, drops=(), in_thinking=False):
    """What the splitter makes of text arriving in these chunks, with neighbouring pieces of one kind joined."""
    split, out = eng.Splitter(dialect, drops, in_thinking), []
    for chunk in [*chunks, None]:
        for kind, text in split.feed(chunk or "", final=chunk is None):
            if out and out[-1][0] == kind:
                out[-1] = (kind, out[-1][1] + text)
            else:
                out.append((kind, text))
    return out


GEMMA = eng.DIALECTS[0]
THINK = eng.DIALECTS[1]


def test_gemma4_thinking_and_answer_are_told_apart_wherever_the_chunks_break():
    text = "<|channel>thought\nHmm, a fox.<channel|>A red fox runs.<turn|>"
    want = [("thinking", "Hmm, a fox."), ("answer", "A red fox runs.")]
    assert pieces(GEMMA, [text], drops=("<turn|>",)) == want
    for cut in range(1, len(text)):
        assert pieces(GEMMA, [text[:cut], text[cut:]], drops=("<turn|>",)) == want, cut          # a marker split in two still counts
    assert pieces(GEMMA, ["<|channel>thought\n<channel|>Just this."]) == [("answer", "Just this.")]      # the empty thought a model writes when thinking is off


def test_think_tags_and_a_prompt_that_already_opened_the_thinking():
    assert pieces(THINK, ["<think>so</think>done"]) == [("thinking", "so"), ("answer", "done")]
    assert pieces(THINK, ["so far</think>done"], in_thinking=True) == [("thinking", "so far"), ("answer", "done")]       # Qwen 3 opens it in the prompt
    assert pieces(eng.DIALECTS[-1], ["no markers <think> here"]) == [("answer", "no markers <think> here")]             # a plain model's text is the answer


def test_the_dialect_follows_the_models_own_vocabulary_or_template():
    assert eng.dialect_for({"<|channel>": 1, "<channel|>": 2}).name == "gemma4"
    assert eng.dialect_for({"<think>": 1, "</think>": 2}).name == "think-tags"
    assert eng.dialect_for({"hello": 1}, "{{ '<tool_call>' }} ... {{ '</tool_call>' }}").name == "tool-tags"
    assert eng.dialect_for({"hello": 1}).name == "plain"


def test_tool_calls_are_read_in_gemma_and_json_form():
    call = eng.parse_call('call:edit_prompt{field:<|"|>chat_prompt<|"|>,new:<|"|>Be short, "always".<|"|>,keep:true,n:3,items:[<|"|>a<|"|>,<|"|>b<|"|>]}')
    assert call == {"name": "edit_prompt", "args": {"field": "chat_prompt", "new": 'Be short, "always".', "keep": True, "n": 3, "items": ["a", "b"]}}
    assert eng.parse_call("not a call") is None
    assert eng._json_call('{"name": "edit_prompt", "arguments": {"field": "x"}}') == {"name": "edit_prompt", "args": {"field": "x"}}
    assert eng._json_call("{broken") is None
    assert [c["name"] for c in pieces_calls(GEMMA, '<|tool_call>call:a{}<tool_call|>x<|tool_call>call:b{x:1}<tool_call|>')] == ["a", "b"]


def pieces_calls(dialect, text):
    split, out, buf = eng.Splitter(dialect), [], ""
    for kind, part in split.feed(text, final=True):
        if kind == "call":
            buf += part
        elif kind == "call_end":
            out.append(dialect.parse(buf))
            buf = ""
    return out


# --- a tiny model, to run the real streaming path -----------------------------

WORDS = ["<pad>", "<bos>", "<eos>", "<unk>", "<think>", "</think>", *"a b c d e f g h i j k l m n o p".split()]
TEMPLATE = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}model: {% if enable_thinking %}<think> {% endif %}{% endif %}")


@pytest.fixture
def tiny():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    torch.manual_seed(0)
    core = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>", chat_template=TEMPLATE)
    cfg = Gemma4TextConfig(vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
                           intermediate_size=64, hidden_size_per_layer_input=8, layer_types=["sliding_attention", "full_attention"], sliding_window=16,
                           max_position_embeddings=256, num_kv_shared_layers=0, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    engine = eng.Engine()
    engine.use(Gemma4ForCausalLM(cfg).eval(), tok, name="tiny")
    return engine


def test_a_model_streams_through_the_engine_piece_by_piece(tiny):
    events = list(tiny.stream([{"role": "user", "content": "a b c"}], max_new_tokens=12, greedy=True))
    assert events[-1] == {"kind": "end", "calls": []}
    assert all(e["kind"] in ("thinking", "answer") and e["text"] for e in events[:-1])
    again = list(tiny.stream([{"role": "user", "content": "a b c"}], max_new_tokens=12, greedy=True))
    assert events == again                                                              # greedy: the same words every time
    seeded = [list(tiny.stream([{"role": "user", "content": "a"}], max_new_tokens=8, seed=7)) for _ in range(2)]
    assert seeded[0] == seeded[1]                                                       # the same seed repeats a sampled answer too


def test_the_enhancer_asks_the_engine_through_a_clip_stand_in_and_hears_each_piece(tiny):
    heard = []
    clip = eng.EngineClip(tiny, heard.append)
    raw, status = clip.run("rules here", "a b", rounds=[("c d", "e f")], max_length=10, do_sample=False)
    assert isinstance(raw, str) and "(engine)" in status and heard[-1]["kind"] == "end"
    assert "".join(e["text"] for e in heard if e["kind"] == "answer") in raw.replace("</think>", "").replace("<think>", "")
    off = eng.Engine()
    assert eng.EngineClip(off).run("s", "u")[0] == "" and "not loaded" in eng.EngineClip(off).run("s", "u")[1]          # no model: said, never raised


def test_a_chat_answer_can_come_from_the_engine(tiny, comfyui, monkeypatch):
    from modules.conditioning.prompt_enhancer import nodes
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    nodes.CONVOS.clear()
    heard = []
    out = nodes.chat_reply("e:0", "a b c", engine_model="tiny", max_length=10, do_sample=False, on_event=heard.append)
    assert "ok" in out and out["status"].endswith("") and heard and heard[-1]["kind"] == "end"
    assert nodes.RUNS[-1]["kind"] == "chat" and nodes.RUNS[-1]["conversation"] == "e:0"
    assert not nodes.chat_reply("e:0", "hi", engine_model="")["ok"]                     # neither an engine model nor an encoder: said


def test_the_rewrite_can_come_from_the_engine_and_a_missing_model_hands_the_prompt_on(tiny, comfyui, monkeypatch):
    from modules.conditioning.prompt_enhancer import nodes
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    nodes.RUNS.clear()
    asked = []
    real = tiny.stream
    monkeypatch.setattr(tiny, "stream", lambda *a, **k: (asked.append(a[0]), real(*a, **k))[1])
    out = nodes.FunUIEnhancePrompt.execute(None, "a b c", enabled=True, engine_model="tiny", max_length=10, greedy=True)
    assert asked and asked[0][-1]["content"].startswith("a b c") and nodes.RUNS[-1]["before"] == "a b c"
    assert out.args[0]                                                                   # whatever the tiny model wrote, a prompt comes out
    missing = nodes.FunUIEnhancePrompt.execute(None, "a fox", enabled=True, engine_model="not/a model", max_length=10)
    assert missing.args[0] == "a fox" and "could not load the engine model" in nodes.RUNS[-1]["status"]
