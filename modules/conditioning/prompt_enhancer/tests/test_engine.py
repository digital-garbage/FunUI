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
    assert events[-1]["kind"] == "end" and events[-1]["calls"] == [] and events[-1]["tokens"] > 0 and "tps" in events[-1]
    assert all(e["kind"] in ("thinking", "answer") and e["text"] for e in events[:-1])
    again = list(tiny.stream([{"role": "user", "content": "a b c"}], max_new_tokens=12, greedy=True))
    assert [e for e in events if e['kind'] != 'end'] == [e for e in again if e['kind'] != 'end']                                                              # greedy: the same words every time
    seeded = [list(tiny.stream([{"role": "user", "content": "a"}], max_new_tokens=8, seed=7)) for _ in range(2)]
    assert seeded[0][:-1] == seeded[1][:-1]                                                       # the same seed repeats a sampled answer too


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


# --- a single-file fp8 checkpoint ----------------------------------------------

def test_a_scaled_fp8_file_runs_like_the_same_weights_widened(tmp_path):
    import torch
    from safetensors.torch import save_file
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from modules.conditioning.prompt_enhancer import fp8
    torch.manual_seed(1)
    cfg = Gemma4TextConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
                           intermediate_size=64, hidden_size_per_layer_input=8, layer_types=["sliding_attention", "full_attention"], sliding_window=16,
                           max_position_embeddings=256, num_kv_shared_layers=0, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg.save_pretrained(tmp_path)
    reference = Gemma4ForCausalLM(cfg).eval().to(torch.bfloat16)
    tensors = {}
    for name, value in reference.state_dict().items():
        if name == "lm_head.weight":
            continue
        big = value.ndim == 2 and ("mlp" in name or "embed_tokens" in name)
        if big:
            scale = value.abs().max().float() / 400
            packed = (value.float() / scale).to(torch.float8_e4m3fn)
            tensors[name] = packed
            tensors[name[:-len("weight")] + "weight_scale"] = scale.reshape(())
            tensors[name[:-len("weight")] + "comfy_quant"] = torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)
            value.copy_((packed.float() * scale).to(torch.bfloat16))          # the reference holds what the file means
        else:
            tensors[name] = value
    save_file(tensors, str(tmp_path / "w.safetensors"))
    assert fp8.is_scaled_fp8(tmp_path / "w.safetensors")
    model = fp8.build(tmp_path, tmp_path / "w.safetensors")
    ids = torch.tensor([[1, 5, 9, 12, 7]])
    with torch.inference_mode():
        got, want = model(ids).logits.float(), reference(ids).logits.float()
    assert torch.allclose(got, want, atol=0.05, rtol=0.05)
    assert next(m for m in model.modules() if isinstance(m, fp8.FP8Linear)).weight.dtype == torch.float8_e4m3fn      # still fp8 in memory


def test_an_engine_source_may_name_a_file_and_a_base():
    import re
    pattern = r"([\w.\-]+/[\w.\-]+)(?:::([^@]+))?(?:@([\w.\-]+/[\w.\-]+))?"
    assert re.fullmatch(pattern, "a/b::x/y.safetensors@c/d").groups() == ("a/b", "x/y.safetensors", "c/d")
    assert eng.local_dir("a/b::f.safetensors@c/d") == eng.local_dir("a/b")


def test_the_speed_ups_write_the_same_words_as_the_plain_way_and_the_cache_is_reused(tiny):
    prompt = [{"role": "user", "content": "a b c d e f a b c d e f a b c"}]
    plain = lambda **k: "".join(e.get("text", "") for e in tiny.stream(prompt, max_new_tokens=14, greedy=True, **k))     # noqa: E731
    base = plain(cache=False, lookup=0)
    assert plain(cache=False, lookup=4) == base                     # prompt lookup changes the speed, never the words
    assert plain(cache=True, lookup=0) == base and tiny._snap is not None
    again = list(tiny.stream(prompt + [{"role": "user", "content": "g h"}], max_new_tokens=14, greedy=True, lookup=0))
    assert again[-1]["cached"] > 0                                  # the shared start of the two requests was not read twice
    other = list(tiny.stream([{"role": "user", "content": "p o n"}], max_new_tokens=6, greedy=True))
    assert other[-1]["cached"] == 0                                 # a request that begins differently starts fresh
    assert plain(cache=True, lookup=4) == base


def test_two_requests_at_once_take_turns_on_the_engine(tiny):
    import threading
    order, done = [], threading.Event()

    def ask(tag):
        for e in tiny.stream([{"role": "user", "content": "a b"}], max_new_tokens=6, greedy=True):
            order.append(tag)
    t1, t2 = threading.Thread(target=ask, args=("x",)), threading.Thread(target=ask, args=("y",))
    t1.start(); t2.start(); t1.join(); t2.join()
    flat = "".join(order)
    assert flat in (flat[0] * flat.count(flat[0]) + flat[-1] * flat.count(flat[-1]),) and set(flat) == {"x", "y"}     # never interleaved


# --- the model editing its own rules --------------------------------------------

def test_a_rule_change_waits_for_the_person_and_the_model_is_told_the_answer():
    import threading
    from modules.conditioning.prompt_enhancer import rules as rl
    sent = []
    editor = rl.Editor({"chat_prompt": "Be chatty.", "notes_prompt": "Notes."}, sent.append, wait=5)
    assert editor({"name": "read_prompt", "args": {"field": "chat_prompt"}}) == "Be chatty."
    assert editor({"name": "read_prompt", "args": {"field": "nope"}}).startswith("Error: there is no rule called 'nope'")
    assert "already" in editor({"name": "edit_prompt", "args": {"field": "chat_prompt", "new_text": "Be chatty.", "reason": "x"}})

    def answer(allow):
        while not sent:
            pass
        ev = sent.pop()
        assert ev["kind"] == "edit_request" and ev["field"] == "chat_prompt" and ev["old"] == "Be chatty." and ev["new"] == "Be short." and ev["reason"] == "too long"
        assert rl.decide(ev["id"], allow)
    for allow, word in ((False, "Disallowed"), (True, "Allowed")):
        t = threading.Thread(target=answer, args=(allow,))
        t.start()
        told = editor({"name": "edit_prompt", "args": {"field": "chat_prompt", "new_text": "Be short.", "reason": "too long"}})
        t.join()
        assert told.startswith(word)
    assert editor.rules["chat_prompt"] == "Be short."            # only the allowed change took hold
    assert rl.Editor({"chat_prompt": "a"}, lambda e: None, wait=0.05)({"name": "edit_prompt", "args": {"field": "chat_prompt", "new_text": "b", "reason": "r"}}).startswith("No answer")
    assert not rl.decide("edit-999", True)


def test_the_model_hears_the_tool_answer_and_carries_on():
    from modules.conditioning.prompt_enhancer import rules as rl

    class Fake:
        device, seen = "cpu", []

        def stream(self, messages, **kw):
            Fake.seen.append([dict(m) for m in messages])
            if len(Fake.seen) == 1:
                yield {"kind": "answer", "text": "Let me check. "}
                yield {"kind": "call", "call": {"name": "read_prompt", "args": {"field": "chat_prompt"}}}
                yield {"kind": "end", "calls": [{"name": "read_prompt", "args": {"field": "chat_prompt"}}], "tps": 1, "cached": 0}
            else:
                yield {"kind": "answer", "text": "It says be chatty."}
                yield {"kind": "end", "calls": [], "tps": 1, "cached": 0}
    heard = []
    clip = eng.EngineClip(Fake(), heard.append, tools=rl.tools(), handler=rl.Editor({"chat_prompt": "Be chatty."}, heard.append))
    raw, status = clip.run("sys", "what is your chat rule?", max_length=20)
    assert "Let me check." in raw and "It says be chatty." in raw and "tokens/s" in status
    second = Fake.seen[1]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"]["name"] == "read_prompt"
    assert second[-1] == {"role": "tool", "tool_call_id": "call0_0", "name": "read_prompt", "content": "Be chatty."}
    assert any(e["kind"] == "tool_result" for e in heard)


def test_the_real_chat_template_renders_a_tool_round_trip():
    pytest.importorskip("jinja2")
    import json, os
    from transformers import AutoTokenizer
    root = os.path.expanduser("~/.cache/funui_llm/Bedovyy__gemma-4-E4B-it-ultra-uncensored-heretic-Comfy/base")
    if not os.path.exists(os.path.join(root, "chat_template.jinja")):
        pytest.skip("the Gemma 4 template is not on this machine")
    from modules.conditioning.prompt_enhancer import rules as rl
    tok = AutoTokenizer.from_pretrained(root)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "read_prompt", "arguments": {"field": "chat_prompt"}}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "read_prompt", "content": "Be chatty."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=rl.tools())
    assert "call:read_prompt" in text and "Be chatty." in text and "edit_prompt" in text


def test_a_model_without_tool_calls_is_told_it_cannot_edit_its_rules_and_still_chats(tiny, comfyui, monkeypatch):
    from modules.conditioning.prompt_enhancer import nodes
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    nodes.CONVOS.clear()
    heard = []
    out = nodes.chat_reply("e:9", "a b c", engine_model="tiny", max_length=8, do_sample=False, on_event=heard.append, self_edit=True, rules={"chat_prompt": "Be chatty."})
    assert "ok" in out and any(e["kind"] == "status" and "cannot edit its own rules" in e["text"] for e in heard)


def test_choosing_another_engine_model_really_loads_it(tiny, monkeypatch):
    asked = []
    monkeypatch.setattr(tiny, "_fetch", lambda source: (asked.append(source), (_ for _ in ()).throw(ValueError("stop here")))[1])
    assert tiny.load_async("other/model") is True
    tiny._worker.join(5)
    assert asked == ["other/model"] and tiny.phase == "error"            # not skipped because the name was already set for the status
    assert tiny.status()["source"] == "other/model"


def test_a_reader_that_goes_away_stops_the_writing_before_the_next_request(tiny):
    import threading
    gen = tiny.stream([{"role": "user", "content": "a b c"}], max_new_tokens=200, greedy=True)
    next(gen)
    gen.close()
    assert not [t for t in threading.enumerate() if t.name == "funui-engine-generate" and t.is_alive()]
    assert tiny.turn.acquire(blocking=False)                              # the turn was given back
    tiny.turn.release()


def test_a_model_with_biases_loads_from_an_fp8_file_in_any_order(tmp_path):
    """Qwen-style attention biases: the filled bias survives whichever order the file's tensors are read in."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from modules.conditioning.prompt_enhancer import fp8
    torch.manual_seed(2)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    cfg.save_pretrained(tmp_path)
    reference = Qwen2ForCausalLM(cfg).eval().to(torch.bfloat16)
    tensors = {}
    for name, value in reference.state_dict().items():
        if name == "lm_head.weight":
            continue
        if value.ndim == 2 and ("proj" in name or "embed_tokens" in name):
            scale = value.abs().max().float() / 400
            packed = (value.float() / scale).to(torch.float8_e4m3fn)
            tensors[name], tensors[name[:-6] + "weight_scale"] = packed, scale.reshape(())
            tensors[name[:-6] + "comfy_quant"] = torch.tensor(list(b"{}"), dtype=torch.uint8)
            value.copy_((packed.float() * scale).to(torch.bfloat16))
        else:
            tensors[name] = value
    assert any(k.endswith("q_proj.bias") for k in tensors)
    save_file(dict(reversed(list(tensors.items()))), str(tmp_path / "w.safetensors"))
    model = fp8.build(tmp_path, tmp_path / "w.safetensors")
    ids = torch.tensor([[1, 5, 9, 12]])
    with torch.inference_mode():
        assert torch.allclose(model(ids).logits.float(), reference(ids).logits.float(), atol=0.05, rtol=0.05)


def test_a_comfy_encoder_file_and_one_file_of_a_repo_are_found_by_name(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import huggingface_hub
    f = tmp_path / "gemma-4-E4B-heretic-fp8.safetensors"
    f.write_bytes(b"x")
    assert eng.comfy_file(str(f)) == str(f)
    with pytest.raises(ValueError, match="not in ComfyUI's text encoders list"):
        eng.comfy_file("nothing-like-this.safetensors")
    repos = {"a/gemma-4-E4B-heretic": ["config.json", "model.safetensors"], "b/gemma-4-E4B-heretic-GGUF": ["x.gguf"], "c/Other-Gemma-4-E4B-heretic": ["config.json"]}

    class Api:
        def list_models(self, search, limit):
            return [NS(id=i) for i in repos]

        def model_info(self, mid):
            return NS(siblings=[NS(rfilename=n) for n in repos[mid]])
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert eng.guess_base("gemma-4-E4B-heretic-fp8.safetensors") == "a/gemma-4-E4B-heretic"        # quantization words in the file name do not count, a loosely similar repo is not taken
    with pytest.raises(ValueError, match="name the base model"):
        eng.guess_base("qwen_3_4b_fp8_scaled.safetensors")


def _tiny_fp8(tmp_path, tie, fp8_embed, own_head):
    import torch
    from safetensors.torch import save_file
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    torch.manual_seed(3)
    cfg = Gemma4TextConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
                           intermediate_size=64, hidden_size_per_layer_input=8, layer_types=["sliding_attention", "full_attention"], sliding_window=16,
                           max_position_embeddings=256, num_kv_shared_layers=0, pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=tie)
    cfg.save_pretrained(tmp_path)
    reference = Gemma4ForCausalLM(cfg).eval().to(torch.bfloat16)
    tensors = {}
    for name, value in reference.state_dict().items():
        embed = name == "model.embed_tokens.weight"
        if name == "lm_head.weight" and not own_head:
            continue
        if value.ndim == 2 and ("mlp" in name or (embed and fp8_embed) or (name == "lm_head.weight" and fp8_embed)):
            scale = value.abs().max().float() / 400
            packed = (value.float() / scale).to(torch.float8_e4m3fn)
            tensors[name], tensors[name[:-6] + "weight_scale"] = packed, scale.reshape(())
            tensors[name[:-6] + "comfy_quant"] = torch.tensor(list(b"{}"), dtype=torch.uint8)
            value.copy_((packed.float() * scale).to(torch.bfloat16))
        else:
            tensors[name] = value
    save_file(tensors, str(tmp_path / "w.safetensors"))
    return reference


@pytest.mark.parametrize("tie,fp8_embed,own_head", [(True, False, False), (False, False, True), (False, True, True), (True, True, False)])
def test_the_output_layer_is_found_whether_tied_or_stored_of_its_own(tmp_path, tie, fp8_embed, own_head):
    import torch
    from modules.conditioning.prompt_enhancer import fp8
    reference = _tiny_fp8(tmp_path, tie, fp8_embed, own_head)
    model = fp8.build(tmp_path, tmp_path / "w.safetensors")
    ids = torch.tensor([[1, 5, 9, 12]])
    with torch.inference_mode():
        assert torch.allclose(model(ids).logits.float(), reference(ids).logits.float(), atol=0.05, rtol=0.05)
