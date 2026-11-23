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
    assert all(e["kind"] in ("thinking", "answer", "status") and e["text"] for e in events[:-1]) and events[0]["kind"] == "status" and "reading the prompt" in events[0]["text"]
    again = list(tiny.stream([{"role": "user", "content": "a b c"}], max_new_tokens=12, greedy=True))
    assert [e for e in events if e['kind'] not in ('end', 'status')] == [e for e in again if e['kind'] not in ('end', 'status')]                                                              # greedy: the same words every time
    seeded = [list(tiny.stream([{"role": "user", "content": "a"}], max_new_tokens=8, seed=7)) for _ in range(2)]
    assert [e for e in seeded[0] if e["kind"] in ("thinking", "answer")] == [e for e in seeded[1] if e["kind"] in ("thinking", "answer")]                                                       # the same seed repeats a sampled answer too


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
    wide = fp8.build(tmp_path, tmp_path / "w.safetensors", precision="bf16")                  # widened once: same words, no fp8 left
    with torch.inference_mode():
        assert torch.allclose(wide(ids).logits.float(), got, atol=0.05, rtol=0.05)
    assert all(m.weight.dtype == torch.bfloat16 for m in wide.modules() if isinstance(m, fp8.FP8Linear))
    assert next(m for m in fp8.build(tmp_path, tmp_path / "w.safetensors", precision="auto").modules() if isinstance(m, fp8.FP8Linear)).weight.dtype == torch.float8_e4m3fn      # no card here: auto keeps fp8


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
            if not (e["kind"] == "status" and "waiting for the model" in e["text"]):          # the one that waits says so first
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
    assert editor({"name": "read_prompt", "args": {"name": "Chat prompt"}}) == "Be chatty."   # other argument names and spellings
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
    quick = rl.Editor({"chat_prompt": "a"}, lambda e: None, wait=0.05)
    quick({"name": "read_prompt", "args": {"field": "chat_prompt"}})
    assert quick({"name": "edit_prompt", "args": {"field": "chat_prompt", "new_text": "b", "reason": "r"}}).startswith("No answer")
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
    assert "ok" in out and any(e["kind"] == "status" and "proposes a rule change in its answer" in e["text"] for e in heard)


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
    assert tiny.turns.waiting() == 0                                      # the turn was given back


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


def test_downloaded_models_are_listed_and_can_be_deleted_even_the_loaded_one(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    (tmp_path / "a__big").mkdir()
    (tmp_path / "a__big" / "w.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "b__small").mkdir()
    (tmp_path / "b__small" / "c.json").write_text("{}")
    tiny.loaded_dir = str(tmp_path / "a__big")
    assert [(m["name"], m["loaded"]) for m in tiny.downloads()] == [("a__big", True), ("b__small", False)]        # biggest first
    assert [m["source"] for m in tiny.downloads()] == ["a/big", "b/small"]                                  # what to load to use each again
    for bad in ("", "..", "../x", "a__big/w.safetensors", "nope"):
        with pytest.raises(ValueError):
            tiny.delete(bad)
    tiny.delete("a__big")
    assert not (tmp_path / "a__big").exists() and tiny.model is None and tiny.phase == "idle"          # the loaded one was unloaded first
    tiny.delete("b__small")
    assert tiny.downloads() == []


def test_a_rewrite_is_kept_in_live_as_it_forms_for_the_tabs_to_show(tiny):
    clip = eng.EngineClip(tiny, live="rewrite")
    seen = []

    def spy(ev):
        seen.append(dict(eng.LIVE.read()))
    clip.on_event = spy
    raw, _ = clip.run("rules", "a b c", max_length=8, do_sample=False)
    run = eng.LIVE.read()
    assert run["kind"] == "rewrite" and run["done"] and run["label"].startswith("a b c") and run["tps"] > 0
    assert (run["thinking"] + run["answer"]) and run["answer"] in raw.replace("<think>", "").replace("</think>", "")
    assert any(not r["done"] and r["answer"] for r in seen) or any(not r["done"] and r["thinking"] for r in seen)       # it was readable before it was finished
    eng.EngineClip(tiny).run("rules", "d e", max_length=4)                                                          # a run that does not ask for it leaves LIVE alone
    assert eng.LIVE.read()["label"].startswith("a b c")


def test_a_loaded_model_stays_in_memory_across_requests_and_only_a_new_precision_loads_it_again(monkeypatch):
    from .. import engine as eng
    e = eng.Engine()
    e.model, e.source, e.weights, e.precision = object(), "m", True, "auto"
    e.load("m", "auto")
    assert e.model is not None                                   # same model, same precision: nothing is read again
    e.weights = False
    e.load("m", "bf16")
    assert e.model is not None                                   # a model that is not a single fp8 file has no precision to change


def test_a_tokenizer_carried_in_the_weights_file_is_written_out_when_the_folder_has_none(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file
    from .. import engine as eng
    data = json.dumps({"version": "1.0"}).encode()
    save_file({"tokenizer_json": torch.tensor(list(data), dtype=torch.uint8)}, str(tmp_path / "w.safetensors"))
    assert eng.write_embedded_tokenizer(tmp_path / "w.safetensors", tmp_path / "base")
    assert (tmp_path / "base" / "tokenizer.json").read_bytes() == data
    save_file({"x": torch.zeros(1)}, str(tmp_path / "n.safetensors"))
    assert not eng.write_embedded_tokenizer(tmp_path / "n.safetensors", tmp_path / "base2")


class _FakeGGUF:
    """What llama.cpp's backend offers the engine: a chat template, the end markers and the text as it forms."""
    gguf, funui_precision, bos, eos = True, "llama.cpp", "<bos>", "<eos>"
    chat_template, projector = "<|channel>thought<channel|>", ""

    def render(self, messages, tools=None, thinking=False, picture=False):
        return "".join(f"[{m['role']}]{m['content']}" for m in messages) + "[assistant]"

    def text_stream(self, text, *, stats=None, cancel=None, **k):
        for piece in ("<|channel>thought\n", "plan it", "<channel|>", "A fox ", "runs.", "<eos>"):
            yield piece
        stats.update(tokens=6, steps=6, prompt=len(text))


def test_a_gguf_model_streams_thinking_and_answer_through_the_same_events():
    from .. import engine as eng
    e = eng.Engine()
    e.model = e.tok = _FakeGGUF()
    e.source, e.phase = "x.gguf", "ready"
    events = list(e.stream([{"role": "user", "content": "hi"}], thinking=True))
    assert "".join(x["text"] for x in events if x["kind"] == "thinking").strip() == "plan it"
    assert "".join(x["text"] for x in events if x["kind"] == "answer") == "A fox runs."
    assert events[-1]["kind"] == "end" and events[-1]["tokens"] == 6


def test_a_repo_of_gguf_files_asks_for_a_pick_and_several_files_can_be_named(monkeypatch, tmp_path):
    import types
    from .. import engine as eng
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    sib = lambda n: types.SimpleNamespace(rfilename=n, size=10)       # noqa: E731
    info = types.SimpleNamespace(siblings=[sib("m-00001-of-00002.gguf"), sib("m-00002-of-00002.gguf"), sib("README.md")], tags=[])
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo, files_metadata=False: info)
    got = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, local_dir=None: got.append(f) or f)
    e = eng.Engine()
    with __import__("pytest").raises(ValueError, match="pick one"):
        e._fetch("a/m")
    folder, weights = e._fetch("a/m::m-00001-of-00002.gguf|m-00002-of-00002.gguf")
    assert got == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"] and weights.endswith("m-00001-of-00002.gguf")


def test_a_picture_projector_beside_a_gguf_model_is_found(tmp_path):
    from .. import gguf
    (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"x")
    assert gguf.find_projector(tmp_path / "m-Q4_K_M.gguf") == ""
    (tmp_path / "m-mmproj-BF16.gguf").write_bytes(b"x")
    assert gguf.find_projector(tmp_path / "m-Q4_K_M.gguf").endswith("m-mmproj-BF16.gguf")


def test_a_picture_is_turned_into_png_and_the_status_says_when_it_could_not_be_sent():
    import io
    import torch
    from PIL import Image
    from .. import engine as eng, gguf
    red = torch.zeros(2, 8, 8, 3)
    red[..., 0] = 1.0
    shown = Image.open(io.BytesIO(gguf.Backend.png(red)))                  # the first frame of a batch
    assert shown.size == (8, 8) and shown.getpixel((0, 0)) == (255, 0, 0)
    e = eng.Engine()
    e.model = e.tok = _FakeGGUF()
    e.source, e.phase, e.device = "x.gguf", "ready", "cpu"
    assert e.sees() is False                                                # no projector file: the picture cannot go in
    raw, status = eng.EngineClip(e).run("sys", "hi", image=red)
    assert "picture was not sent" in status


def test_a_fresh_llama_install_that_does_not_load_asks_for_a_restart_and_then_falls_back_to_the_processor_build(monkeypatch, tmp_path):
    import importlib.util
    from .. import engine as eng, gguf
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gguf, "wheel_index", lambda: ("https://x/cu124", True))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)                 # not installed yet
    state, calls, said = {"loads": 0}, [], []

    def importer():
        state["loads"] += 1
        if state["loads"] < 4:
            raise RuntimeError("Failed to load shared library 'libllama.so': libcudart.so.12: cannot open shared object file")
        return "module"
    monkeypatch.setattr(gguf, "_import", importer)
    monkeypatch.setattr(gguf, "_pip", lambda args, say, what: calls.append(args) or (True, "ok"))
    with __import__("pytest").raises(RuntimeError, match="Restart ComfyUI"):
        gguf.ensure_llama(said.append)
    assert calls[0][0] == "llama-cpp-python" and calls[1][0] == "nvidia-cuda-runtime-cu12"        # everything is installed together, once
    assert any("restart" in x for x in said)                                                       # and the person is told up front
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())            # after the restart: installed, still failing
    assert gguf.ensure_llama() == "module" and "--force-reinstall" in calls[-1]       # the processor build is the last resort


def test_a_failed_llama_decode_clears_the_memory_and_runs_the_request_again_once():
    from .. import gguf

    class Llm:
        n_tokens = 0
        resets = 0
        fail = True

        def n_ctx(self): return 4096
        def token_eos(self): return 2
        def tokenize(self, raw, add_bos=False, special=True): return [5, 6, 7] if len(raw) > 3 else [9]
        def detokenize(self, ids, special=False): return b"x"
        def set_seed(self, s): pass
        def reset(self): Llm.resets += 1

        class _ctx:
            @staticmethod
            def kv_cache_clear(): pass

        def generate(self, tokens, **k):
            if Llm.fail:
                Llm.fail = False
                raise RuntimeError("llama_decode returned 1")
            yield 11
            yield 2
    backend = gguf.Backend(Llm(), "x.gguf")
    out = "".join(backend.text_stream("hello there", max_new_tokens=5, temperature=0.5, top_p=1, top_k=1, min_p=0, repetition_penalty=1, presence_penalty=0, seed=0, greedy=False))
    assert out == "x" and Llm.resets == 1


def test_a_rule_change_written_into_the_answer_is_found_and_taken_out():
    from .. import rules as rl
    text = "Done, shorter answers from now on.\n\nEDIT_RULE: chat_prompt\nAnswer in one sentence.\nNo lists.\nEND_EDIT_RULE"
    clean, found = rl.parse_edits(text)
    assert clean == "Done, shorter answers from now on."
    assert found == [{"field": "chat_prompt", "new_text": "Answer in one sentence.\nNo lists."}]
    assert rl.parse_edits("nothing to change") == ("nothing to change", [])


def test_thinking_gets_its_own_token_room_on_top_of_the_answer_s(monkeypatch):
    from .. import engine as eng
    e = eng.Engine()
    seen = []
    monkeypatch.setattr(e, "_stream", lambda messages, **o: seen.append(o["max_new_tokens"]) or iter(()))
    list(e.stream([], max_new_tokens=400, thinking=True))
    list(e.stream([], max_new_tokens=400, thinking=False))
    assert seen == [400 + eng.THINK_BUDGET, 400]


def test_a_download_fetches_files_without_touching_the_model_that_is_loaded(monkeypatch, tmp_path):
    from .. import engine as eng
    e = eng.Engine()
    model = object()
    e.model, e.tok, e.source, e.phase, e.message = model, model, "a/b::q4.gguf", "ready", "a/b::q4.gguf on cuda"
    seen = []
    def fake_fetch(source):
        seen.append((source, e.phase))
        e._dir = str(tmp_path)
        return str(tmp_path), ""
    monkeypatch.setattr(e, "_fetch", fake_fetch)
    e.download("a/b::q8.gguf")
    assert seen == [("a/b::q8.gguf", "downloading")]
    assert e.model is model and e.source == "a/b::q4.gguf" and e.phase == "ready"      # still the loaded one
    assert (tmp_path / ".source").read_text() == "a/b::q8.gguf"
    assert e.message == "a/b::q4.gguf on cuda. Downloaded a/b: q8.gguf."


def test_a_model_is_named_short_in_the_status():
    from ..engine import label
    assert label("a/b::x.gguf") == "a/b: x.gguf"
    assert label("a/b::x.gguf|y.gguf|z.gguf@c/d") == "a/b (3 files)"
    assert label("a/b") == "a/b" and label("") == ""


def test_comfy_stop_cancels_a_rewrite_but_not_chat(monkeypatch):
    from .. import engine as eng
    stopped = {"on": False}
    monkeypatch.setattr(eng, "comfy_interrupted", lambda: stopped["on"])
    rewrite, chat = eng.Cancel(True), eng.Cancel(False)
    assert not rewrite.is_set() and not chat.is_set()
    stopped["on"] = True
    assert rewrite.is_set()
    assert not chat.is_set()


def test_a_written_rule_change_without_the_closing_line_still_counts():
    from .. import rules as rl
    clean, found = rl.parse_edits("I will do that.\nEDIT_RULE: chat_prompt\nAnswer in one sentence.")
    assert clean == "I will do that."
    assert found == [{"field": "chat_prompt", "new_text": "Answer in one sentence."}]
    clean, found = rl.parse_edits("A\nEDIT_RULE: chat_prompt\nx\nEND_EDIT_RULE\nthen B")
    assert found == [{"field": "chat_prompt", "new_text": "x"}] and clean.startswith("A") and clean.endswith("then B")


def test_a_helper_that_only_talks_about_a_change_is_told_once_to_call_the_tool(tiny, comfyui, monkeypatch):
    import types
    from modules.conditioning.prompt_enhancer import engine as eng, helper, nodes
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    monkeypatch.setattr(tiny.tok, "chat_template", "{{ tools }}", raising=False)
    monkeypatch.setattr(eng, "dialect_for", lambda *a, **k: types.SimpleNamespace(call_open="<call>"))
    nodes.CONVOS.clear()
    asked = []

    def fake(clip, text, **kw):
        asked.append((text, [t["reply"] for t in kw["convo"]["turns"]]))
        if len(asked) == 1:
            return "I will use the `set_story` tool to merge the scenes.", "ok", [], [], ""
        return "Merged.", "ok", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", fake)
    out = nodes.chat_reply("h:1", "merge the scenes", engine_model="tiny", max_length=8, do_sample=False, on_event=lambda e: None, helper={"sheet": "x", "scenes": 2})
    assert [a[0] for a in asked][:4] == ["merge the scenes", helper.NUDGE, helper.PLAN_NOTE, helper.PLAN_NOTE]          # then the checklist is asked for, twice, as it wrote none
    assert "nothing was changed" in asked[4][0]          # then one round that names the fitting tools
    assert asked[1][1] == ["I will use the `set_story` tool to merge the scenes."]        # it sees what it wrote
    assert out["reply"].startswith("Merged.") and "Nothing was changed" in out["reply"]          # no call came, and the answer says so
    assert [t["reply"] for t in nodes.CONVOS["h:1"]["turns"]] == [out["reply"]]                # the conversation keeps the final answer only


def test_naming_one_part_of_a_split_gguf_model_downloads_every_part(monkeypatch, tmp_path):
    import types
    import huggingface_hub
    from .. import engine as eng
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    sib = lambda n: types.SimpleNamespace(rfilename=n, size=10)       # noqa: E731
    info = types.SimpleNamespace(siblings=[sib(f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)], tags=[])
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo, files_metadata=False: info)
    got = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, local_dir=None: got.append(f) or f)
    folder, weights = eng.Engine()._fetch("a/m::m-00002-of-00003.gguf")
    assert got == ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"] and weights.endswith("m-00001-of-00003.gguf")


def test_a_missing_part_of_a_split_model_is_named():
    from .. import engine as eng
    assert eng.shard_set("model-00002-of-00003.safetensors", {"model-00001-of-00003.safetensors": 1, "model-00002-of-00003.safetensors": 1, "model-00003-of-00003.safetensors": 1}) == [
        f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    assert eng.shard_set("model.safetensors", {}) == []
    with __import__("pytest").raises(ValueError, match="no part model-00002-of-00003"):
        eng.shard_set("model-00001-of-00003.safetensors", {"model-00001-of-00003.safetensors": 1})
    assert eng.shard_index("model-00001-of-00002.safetensors", {"model.safetensors.index.json": 1}) == "model.safetensors.index.json"
    assert eng.shard_index("x-00001-of-00002.safetensors", {}) == ""


def test_naming_one_safetensors_part_downloads_all_parts_the_index_and_the_config_into_one_folder(monkeypatch, tmp_path):
    import types
    import huggingface_hub
    from .. import engine as eng
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    names = ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    sib = lambda n: types.SimpleNamespace(rfilename=n, size=10)       # noqa: E731
    info = types.SimpleNamespace(siblings=[sib(n) for n in names], tags=[])
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo, files_metadata=False: info)
    got, snaps = [], []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, local_dir=None: got.append(f) or f)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, local_dir=None, **kw: snaps.append((repo, local_dir)))
    folder, weights = eng.Engine()._fetch("a/m::model-00002-of-00002.safetensors")
    assert got == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"]
    assert weights == "" and snaps[0][0] == "a/m" and snaps[0][1] == folder        # read as an ordinary checkpoint folder


def test_a_helper_model_without_tool_calls_works_through_tool_blocks_in_its_answer(tiny, comfyui, monkeypatch):
    import threading
    from modules.conditioning.prompt_enhancer import helper, nodes, rules
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    monkeypatch.setattr(tiny.tok, "chat_template", "plain chat", raising=False)
    nodes.CONVOS.clear()
    asked, heard = [], []

    def on_event(ev):
        heard.append(ev)
        if ev["kind"] == "action_request":
            threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()

    def fake(clip, text, **kw):
        asked.append(text)
        assert "TOOL: set_clip_length" in kw["system_extra"]                                  # the text-mode instructions went in
        if len(asked) == 1:
            return 'TOOL: set_clip_length\n{"seconds": 6, "reason": "asked"}\nEND_TOOL', "ok", [], [], ""
        return "Clip length is 6 seconds now.", "ok", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", fake)
    out = nodes.chat_reply("h:2", "make clips 6 seconds", engine_model="tiny", max_length=8, do_sample=False, on_event=on_event, helper={"sheet": "x", "scenes": 1})
    assert out["reply"] == "Clip length is 6 seconds now." and asked[1].startswith("TOOL RESULTS:\nset_clip_length: Allowed")
    assert any(e["kind"] == "action_request" and e["tool"] == "set_clip_length" for e in heard)


def test_a_helper_that_answers_without_any_call_is_asked_again_in_text_mode(tiny, comfyui, monkeypatch):
    import threading
    from modules.conditioning.prompt_enhancer import nodes, rules
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    monkeypatch.setattr(tiny.tok, "chat_template", "tools {{ tools }}", raising=False)
    from modules.conditioning.prompt_enhancer import engine as eng
    monkeypatch.setattr(eng, "dialect_for", lambda *a, **k: type("D", (), {"call_open": "<c>"})())
    nodes.CONVOS.clear()
    asked = []

    def on_event(ev):
        if ev["kind"] == "action_request":
            threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()

    def fake(clip, text, **kw):
        asked.append(text)
        if len(asked) == 1:
            return "Sure, one scene on the face then.", "ok", [], [], ""          # no call and nothing that reads as an announcement
        if len(asked) == 2:
            return 'TOOL: set_clip_length\n{"seconds": 6, "reason": "asked"}\nEND_TOOL', "ok", [], [], ""
        return "Done, six seconds.", "ok", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", fake)
    out = nodes.chat_reply("h:3", "make it one scene focused on the face", engine_model="tiny", max_length=8, do_sample=False, on_event=on_event, helper={"sheet": "x", "scenes": 3})
    assert out["reply"] == "Done, six seconds." and asked[2].startswith("TOOL RESULTS:")


def test_a_gguf_model_that_cannot_trim_its_memory_starts_empty_for_a_changed_prompt():
    from .. import gguf
    calls = []

    class Llm:
        n_tokens = 3
        input_ids = __import__("numpy").array([1, 2, 3])
        def reset(self): calls.append("reset")                                        # noqa: E704
        def tokenize(self, *a, **k): return [1, 9, 9]                                  # noqa: E704
        def token_eos(self): return 0                                                  # noqa: E704
        def n_ctx(self): return 1000                                                   # noqa: E704
        def set_seed(self, s): pass                                                    # noqa: E704
        def generate(self, tokens, reset=True, **k): return iter(())                   # noqa: E704
        _ctx = type("C", (), {"kv_cache_clear": lambda self: calls.append("clear")})()
    b = gguf.Backend.__new__(gguf.Backend)
    b.llm = Llm()
    b.can_trim = lambda: False
    list(b.text_stream("x", max_new_tokens=4, temperature=0, top_p=1, top_k=1, min_p=0, repetition_penalty=1, presence_penalty=0, seed=0, greedy=True))
    assert calls == ["reset", "clear"]


def test_a_helper_that_acts_and_then_says_nothing_is_not_a_failure(tiny, comfyui, monkeypatch):
    import threading
    from modules.conditioning.prompt_enhancer import nodes, rules
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    monkeypatch.setattr(tiny.tok, "chat_template", "plain chat", raising=False)
    nodes.CONVOS.clear()
    asked = []

    def on_event(ev):
        if ev["kind"] == "action_request":
            threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()

    def fake(clip, text, **kw):
        asked.append(text)
        if len(asked) == 1:
            return 'TOOL: set_clip_length\n{"seconds": 6, "reason": "asked"}\nEND_TOOL', "ok", [], [], ""
        return "", "the model returned nothing", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", fake)
    out = nodes.chat_reply("h:4", "make clips 6 seconds", engine_model="tiny", max_length=8, do_sample=False, on_event=on_event, helper={"sheet": "x", "scenes": 1})
    assert out["ok"] and out["reply"].startswith("Done.")


def test_several_pictures_each_get_a_placeholder_and_a_bitmap():
    from .. import gguf
    b = gguf.Backend.__new__(gguf.Backend)
    b.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    b.marker = lambda: "<__media__>"
    b.llm = type("L", (), {"token_bos": lambda self: -1, "token_eos": lambda self: -1})()
    text = b.render([{"role": "user", "content": "describe"}], picture=3)
    assert text.count("<__media__>") == 3 and text.rstrip().endswith("describe")
    np = __import__("numpy")
    frames = [np.zeros((1, 4, 4, 3)), np.ones((2, 4, 4, 3))]
    assert len(gguf.Backend.pngs(frames)) == 2 and len(gguf.Backend.pngs(frames[0])) == 1


def test_a_chat_that_starts_during_a_rewrite_neither_takes_over_nor_ends_the_rewrites_live_view():
    live = eng.Live()
    rewrite = live.start("rewrite", "a prompt")
    live.feed({"kind": "thinking", "text": "hmm "}, rewrite)
    chat = live.start("chat", "a question")                      # asked while the rewrite is still writing
    live.feed({"kind": "answer", "text": "the rewrite goes on"}, rewrite)
    live.feed({"kind": "end", "tps": 5.0}, chat)                  # the chat's own end must not end the rewrite
    now = live.read()
    assert now["kind"] == "rewrite" and not now["done"] and now["answer"] == "the rewrite goes on" and now["thinking"] == "hmm "
    live.feed({"kind": "end", "tps": 7.0}, rewrite)
    assert live.read()["done"] and live.read()["tps"] == 7.0


def test_requests_for_the_model_are_served_first_come_first_served_and_say_they_wait():
    import threading
    turns = eng.Turns()
    first, ahead, _ = turns.take()
    assert ahead == 0
    turns.wait(first, "a rewrite")
    second, ahead2, doing = turns.take()
    third, ahead3, _ = turns.take()
    assert (ahead2, ahead3, doing) == (1, 2, "a rewrite")
    order = []

    def go(place, name):
        turns.wait(place, name)
        order.append(name)
        turns.release(place)
    threads = [threading.Thread(target=go, args=(third, "third")), threading.Thread(target=go, args=(second, "second"))]
    [t.start() for t in threads]
    import time
    time.sleep(0.05)
    assert order == []                                    # both wait for the first
    turns.release(first)
    [t.join(2) for t in threads]
    assert order == ["second", "third"] and turns.waiting() == 0


def test_a_request_that_gives_up_before_its_turn_does_not_block_the_ones_behind_it():
    turns = eng.Turns()
    a, _, _ = turns.take()
    b, _, _ = turns.take()
    c, _, _ = turns.take()
    turns.wait(a)
    turns.release(b)                                      # the page went away while b was queued
    turns.release(a)
    turns.wait(c)                                         # c is next, not stuck behind b
    turns.release(c)
    assert turns.waiting() == 0


def test_a_model_change_waits_for_the_request_that_is_writing(tiny):
    import threading
    import time
    place, _, _ = tiny.turns.take()
    tiny.turns.wait(place, "a rewrite")
    done = []
    threading.Thread(target=lambda: (tiny.unload(), done.append(1)), daemon=True).start()
    time.sleep(0.1)
    assert not done                                       # the model stays while the rewrite writes
    tiny.turns.release(place)
    for _ in range(50):
        if done:
            break
        time.sleep(0.02)
    assert done and tiny.model is None


def test_the_reported_speed_counts_the_writing_only_not_the_time_to_read_the_prompt():
    import time
    from .. import engine as eng

    class Slow(_FakeGGUF):
        def text_stream(self, text, *, stats=None, **kw):
            time.sleep(0.3)                                        # reading a long prompt
            for i, piece in enumerate(["A ", "fox ", "runs", "."]):
                if stats is not None:
                    stats["prompt"], stats["steps"] = 900, i + 1
                    stats["last"] = time.time()
                    if i == 0:
                        stats["first"] = stats["last"]
                time.sleep(0.05)
                yield piece
            if stats is not None:
                stats["tokens"], stats["cached"] = 4, 0
    e = eng.Engine()
    e.model = e.tok = Slow()
    e.source, e.phase = "x.gguf", "ready"
    end = list(e.stream([{"role": "user", "content": "hi"}]))[-1]
    assert end["kind"] == "end" and end["read_seconds"] >= 0.25
    assert end["tps"] > 10                                         # 3 tokens over about 0.15 s, not 4 over 0.5 s


def test_a_session_that_ended_abruptly_with_llama_cpp_loaded_is_reported_with_its_last_native_lines(monkeypatch, tmp_path):
    import os
    import types
    from .. import engine as eng, gguf
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gguf, "_KEEP", [])
    (tmp_path / ".llama_running").write_text("999999")                      # a process that is gone
    (tmp_path / "llama_native.log").write_text("load ok\nGGML_ASSERT(found) failed\n")
    hooked = []
    fake = types.SimpleNamespace(llama_log_callback=lambda f: f, llama_log_set=lambda cb, data: hooked.append(cb))
    said = []
    gguf.watch_native(fake, said.append)
    assert said and "ended abruptly" in said[0] and "GGML_ASSERT(found) failed" in said[0]
    assert (tmp_path / ".llama_running").read_text() == str(os.getpid())    # this process now holds the marker
    hooked[0](2, b"ggml_cuda_init: found 1 CUDA devices\n", None)
    assert "found 1 CUDA devices" in (tmp_path / "llama_native.log").read_text()


def test_the_gguf_model_is_loaded_with_llama_cpps_own_batch_sizes(monkeypatch, tmp_path):
    """A picture is decoded in one piece that must fit the micro-batch: larger batch sizes than the default made llama.cpp abort the whole ComfyUI process on a rewrite with a picture."""
    import types
    from .. import engine as eng, gguf
    monkeypatch.setattr(eng, "models_dir", lambda: str(tmp_path))
    seen = {}
    fake = types.SimpleNamespace(Llama=lambda **kw: seen.update(kw) or types.SimpleNamespace(), llama_supports_gpu_offload=lambda: True,
                                 llama_log_callback=lambda f: f, llama_log_set=lambda *a: None)
    monkeypatch.setattr(gguf, "ensure_llama", lambda say=None: fake)
    monkeypatch.setattr(gguf, "find_projector", lambda path: "")
    gguf.load(str(tmp_path / "m.gguf"), cuda=True)
    assert "n_batch" not in seen and "n_ubatch" not in seen


def test_a_sliding_window_model_starts_from_empty_instead_of_trimming():
    from modules.conditioning.prompt_enhancer import gguf
    b = gguf.Backend.__new__(gguf.Backend)
    b.llm = type("L", (), {"metadata": {"general.architecture": "gemma4", "gemma4.attention.sliding_window": "1024"}})()
    assert b.can_trim() is False


def test_a_model_that_cannot_trim_resumes_from_a_copy_kept_at_the_last_turn():
    import numpy as np
    from modules.conditioning.prompt_enhancer import gguf
    log = []

    class Llm:
        n_tokens = 0
        input_ids = np.zeros(400, dtype=int)
        _ctx = type("C", (), {"kv_cache_clear": lambda self: log.append("clear")})()
        def reset(self): self.n_tokens = 0; log.append("reset")                       # noqa: E704
        def tokenize(self, raw, add_bos=False, special=True): return [c % 90 + 1 for c in raw]  # noqa: E704
        def eval(self, toks):
            log.append(("eval", len(toks)))
            self.input_ids[self.n_tokens:self.n_tokens + len(toks)] = toks
            self.n_tokens += len(toks)
    b = gguf.Backend.__new__(gguf.Backend)
    b.llm = Llm()
    b._snapshot = lambda ids: b.__dict__.setdefault("snapshots", []).append(tuple(ids))
    b._restore = lambda ids, data: (log.append(("restore", len(ids))), setattr(b.llm, "n_tokens", len(ids)))
    turn = "<|im_start|>assistant\n"
    base = "<|im_start|>system\nabc<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    first = base + turn
    feed, fresh = b._resume(first, b.llm.tokenize(first.encode()))
    assert (len(feed), fresh) == (len(turn), False)                      # all but the last turn was read and kept
    assert len(b.snapshots[0]) == len(base)
    b.llm.n_tokens = len(first) + 5                                      # the answer it wrote sits at the end of its memory now
    b._snapshot = lambda ids: None
    b.snapshots = [(tuple(b.llm.tokenize(base.encode())), b"x")]
    log.clear()
    second = base + turn + "ok<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n" + turn
    feed, fresh = b._resume(second, b.llm.tokenize(second.encode()))
    assert ("restore", len(base)) in log and fresh is False
    assert len(feed) == len(turn) and ("eval", len(second) - len(base) - len(turn)) in log   # the new middle was read, then only the last turn is left


def test_the_reference_intro_forbids_writing_names_and_triggers():
    from modules.conditioning.prompt_enhancer import enhance
    assert "DO NOT OUTPUT THEM" in enhance.REFERENCE_INTRO


def _rule_chat(monkeypatch, tiny, behave, message="Soo.. will you correct them?", before=None):
    import threading
    from modules.conditioning.prompt_enhancer import nodes, rules
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    monkeypatch.setattr(tiny.tok, "chat_template", "plain chat", raising=False)
    nodes.CONVOS.clear()
    if before:
        nodes.CONVOS["h:rules"] = {"summary": "", "turns": __import__("collections").deque([before], maxlen=8), "seen": set()}
    proposals = []

    def on_event(ev):
        if ev["kind"] == "edit_request":
            proposals.append(ev["new"])
            threading.Timer(0.01, lambda: rules.decide(ev["id"], True)).start()
    monkeypatch.setattr(nodes.en, "converse", lambda clip, text, **kw: behave(text))
    out = nodes.chat_reply("h:rules", message, engine_model="tiny", max_length=8, do_sample=False, on_event=on_event,
                           helper={"sheet": "x", "scenes": 1, "rules": {"instructions": "Write 2.5D illustration prompts."}})
    return out, proposals


_EDIT = 'TOOL: edit_prompt\n{"field": "instructions", "new_text": "Write prompts in the requested style.", "reason": "asked"}\nEND_TOOL'


def test_a_request_answered_by_only_looking_at_a_rule_is_pushed_into_changing_it(monkeypatch, tiny, comfyui):
    seen = []

    def behave(text):
        seen.append(text)
        if "nothing was changed" in text:
            return _EDIT, "ok", [], [], ""
        return ('TOOL: read_prompt\n{"field": "instructions"}\nEND_TOOL' if len(seen) == 1 else ""), "ok", [], [], ""
    out, proposals = _rule_chat(monkeypatch, tiny, behave, message="Please correct the rules so they do not mention 2.5D")
    assert proposals == ["Write prompts in the requested style."] and out["ok"]
    assert not out["reply"].startswith("Done. The changes it asked for were handled above") or proposals


def test_a_follow_up_question_after_an_announcement_without_a_call_forces_the_call(monkeypatch, tiny, comfyui):
    before = {"request": "Please correct the rules so they do not mention 2.5D", "reply": "I will remove the mention of 2.5D from the rules.", "kind": "chat"}

    def behave(text):
        return (_EDIT if ("nothing was changed" in text or "Still nothing" in text or "have not read" in text) else "Yes, I will."), "ok", [], [], ""
    out, proposals = _rule_chat(monkeypatch, tiny, behave, before=before)
    assert proposals == ["Write prompts in the requested style."]


def test_a_rule_cannot_be_changed_blind_or_cut_down_to_a_fragment():
    from modules.conditioning.prompt_enhancer import rules as rl
    sent = []
    long = "Rule line. " * 60
    ed = rl.Editor({"instructions": long}, sent.append, wait=0.05)
    told = ed({"name": "edit_prompt", "args": {"field": "instructions", "new_text": "Short.", "reason": "x"}})
    assert told.startswith("Error") and long in told and not sent         # refused, and the model is handed the text
    told = ed({"name": "edit_prompt", "args": {"field": "instructions", "new_text": "Short.", "reason": "x"}})
    assert told.startswith("Error: the new text is much shorter") and not sent
    assert ed({"name": "read_prompt", "args": {}}).startswith("Error") and "instructions" in ed({"name": "read_prompt", "args": {}})


def test_a_plain_question_is_not_forced_into_a_change(monkeypatch, tiny, comfyui):
    out, proposals = _rule_chat(monkeypatch, tiny, lambda text: ("It mentions 2.5D once.", "ok", [], [], ""), message="Does the rule mention 2.5D?")
    assert proposals == [] and out["reply"] == "It mentions 2.5D once."


def test_the_tools_shown_follow_the_request_and_all_come_back_when_it_names_none():
    from modules.conditioning.prompt_enhancer import helper as hp
    names = lambda only: {t["function"]["name"] for t in hp.tools(True, only)}          # noqa: E731
    assert names(hp.intent("update the system prompt")) >= {"read_prompt", "edit_prompt"}
    assert "set_variable" not in names(hp.intent("update the system prompt"))
    assert names(hp.intent("do it")) == names(None)
    assert hp.wants_change("Soo.. will you correct them?", "I will remove the mention.") and not hp.wants_change("Soo.. will you correct them?", "Yes it is fine.")


def test_the_stop_button_ends_a_working_request_and_a_new_message_interrupts_and_follows_it(monkeypatch, tiny, comfyui):
    import threading
    from modules.conditioning.prompt_enhancer import nodes
    monkeypatch.setattr(nodes, "ENGINE", tiny)
    monkeypatch.setattr(tiny.tok, "chat_template", "plain chat", raising=False)
    nodes.CONVOS.clear()
    started, release, seen = threading.Event(), threading.Event(), []

    def fake(clip, text, **kw):
        seen.append((text, [t["request"] + " => " + t["reply"] for t in kw["convo"]["turns"]]))
        if len(seen) == 1:
            started.set()
            release.wait(5)                                              # the first request is still writing
        return "ok", "ok", [], [], ""
    monkeypatch.setattr(nodes.en, "converse", fake)
    results = {}
    first = threading.Thread(target=lambda: results.setdefault("a", nodes.chat_reply("h:s", "first", engine_model="tiny", max_length=8, do_sample=False)))
    first.start()
    assert started.wait(5)
    assert nodes.stop_chat("h:s") is True
    release.set()
    first.join(5)
    assert results["a"]["stopped"] and not results["a"]["ok"]
    assert [t["request"] for t in nodes.CONVOS["h:s"]["turns"]] == ["first"] and "stopped" in nodes.CONVOS["h:s"]["turns"][0]["reply"]
    assert nodes.stop_chat("h:s") is False                               # nothing is running now
    # a new message while one is still working: that one is stopped, and the new one reads it together with the new message
    seen.clear()
    started.clear(), release.clear()
    second = threading.Thread(target=lambda: results.setdefault("b", nodes.chat_reply("h:s", "second", engine_model="tiny", max_length=8, do_sample=False)))
    second.start()
    assert started.wait(5)
    third = threading.Thread(target=lambda: results.setdefault("c", nodes.chat_reply("h:s", "third", engine_model="tiny", max_length=8, do_sample=False)))
    third.start()
    release.set()
    second.join(5), third.join(5)
    assert results["b"]["stopped"] and results["c"]["ok"]
    assert any("second => (the user stopped" in t for t in seen[-1][1])


def test_entries_can_be_sent_without_their_names():
    from modules.conditioning.prompt_enhancer import enhance as en
    group = [{"block": "[Shortcut] Rain\nIt pours.", "bare": "It pours.", "words": ["rain"], "vocab": set()},
             {"block": "[Lore] Castle\nOld stones.", "bare": "Old stones.", "keys": "castle"}]
    named = en.reference("rain", [group])
    bare = en.reference("rain", [group], names=False)
    assert "Rain" in named and "[Shortcut]" in named
    assert "It pours." in bare and "Rain" not in bare and "[Shortcut]" not in bare and "always rewrite" in bare and "name" not in bare.split("It pours")[0].lower()


def test_a_message_that_interrupted_a_change_request_is_forced_into_the_change_too(monkeypatch, tiny, comfyui):
    before = {"request": "Please correct the rules so they do not mention 2.5D", "reply": "(the user stopped this request before it was answered)", "kind": "chat"}

    def behave(text):
        return (_EDIT if ("nothing was changed" in text or "Still nothing" in text or "have not read" in text) else "Understood, noted."), "ok", [], [], ""
    out, proposals = _rule_chat(monkeypatch, tiny, behave, message="Also no 3D look.", before=before)
    assert proposals == ["Write prompts in the requested style."]


def _split(dialect, chunks, **kw):
    from modules.conditioning.prompt_enhancer import engine as eng
    sp = eng.Splitter(next(d for d in eng.DIALECTS if d.name == dialect), **kw)
    got = []
    for ch in chunks:
        got += sp.feed(ch)
    got += sp.feed("", final=True)
    out = {}
    for kind, text in got:
        out[kind] = out.get(kind, "") + text
    return out


def test_gemma_thinking_never_reaches_the_answer_however_the_text_is_cut_up():
    whole = "<|channel>thought\nThe user wants a longer scene.<channel|>Scene 1 is longer now."
    for size in (1, 2, 3, 7, len(whole)):
        out = _split("gemma4", [whole[i:i + size] for i in range(0, len(whole), size)])
        assert out == {"thinking": "The user wants a longer scene.", "answer": "Scene 1 is longer now."}, size


def test_thinking_that_ends_in_a_tool_call_or_never_ends_stays_thinking():
    out = _split("gemma4", ["<|channel>thought\nI should edit scene 1.", "<|tool_call>call:edit_scene{scene:1}<tool_call|>"])
    assert out["thinking"] == "I should edit scene 1." and "answer" not in out and out["call"] == "call:edit_scene{scene:1}"
    cut = _split("gemma4", ["<|channel>thought\nreasoning that runs out of room"])           # the token budget ended it: no answer, and no thinking written as one
    assert cut == {"thinking": "reasoning that runs out of room"}
    qwen = _split("think-tags", ["<think>plan", "</think>Done."])
    assert qwen == {"thinking": "plan", "answer": "Done."}


def test_a_gemma_call_is_read_however_the_model_quotes_its_texts():
    from modules.conditioning.prompt_enhancer import engine as eng
    Q = '<|"|>'
    want = {"name": "edit_shortcuts", "args": {"changes": [{"name": "Arms up (hands high)", "new_name": "Pose, arms up"}], "reason": "r"}}
    for text in (f'call:edit_shortcuts{{changes:[{{name:{Q}Arms up (hands high){Q},new_name:{Q}Pose, arms up{Q}}}],reason:{Q}r{Q}}}',      # the format's own quotes
                 'call:edit_shortcuts{changes:[{name:"Arms up (hands high)",new_name:"Pose, arms up"}],reason:"r"}',                        # plain double quotes
                 "call:edit_shortcuts{changes:[{name:'Arms up (hands high)',new_name:'Pose, arms up'}],reason:'r'}",                        # single quotes
                 'call:edit_shortcuts{changes:[{name:Arms up (hands high),new_name:Pose, arms up}],reason:r}'):                              # no quotes at all
        assert eng.parse_call(text) == want, text
    assert eng.parse_call('call:plan{items:[set a, set b (x), "set c"]}')["args"] == {"items": ["set a", "set b (x)", "set c"]}
    assert eng.parse_call('call:edit_scene{scene:2,seconds:5.5,text:a dog runs, fast,reason:r}')["args"] == {"scene": 2, "seconds": 5.5, "text": "a dog runs, fast", "reason": "r"}


def test_a_call_that_cannot_be_read_is_answered_with_an_error_not_dropped():
    from modules.conditioning.prompt_enhancer import engine as eng, helper
    d = next(x for x in eng.DIALECTS if x.name == "gemma4")
    calls = []
    got = list(eng.Engine._pieces(["<|tool_call>call:edit_shortcuts{changes:[{name:<|\"|>broken", "<tool_call|>"], eng.Splitter(d), calls, d))
    assert calls and calls[0]["name"] == "edit_shortcuts" and calls[0]["unreadable"].startswith("call:edit_shortcuts") and got[0]["kind"] == "call"
    acts = helper.Actions(1, lambda e: None, wait=1)
    told = acts(calls[0])
    assert told.startswith("Error: the call to edit_shortcuts could not be read") and acts.changes == 0 and acts.calls == 1
