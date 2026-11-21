"""The prompt enhancer's own text engine, on transformers.

ComfyUI's text generation answers only when it is finished and offers no say over what the model writes. This engine
loads a Hugging Face model of its own, streams what it writes as it forms (thinking and answer apart), and reads the
tool calls a Gemma 4 model writes. It is optional: with no engine model set, the enhancer keeps using ComfyUI's encoder.

Nothing here imports torch or transformers until a model is asked for, so a machine without them loads the module.
"""

import copy
import os
import random
import re
import threading
import time

REPO_FILES = ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.tiktoken"]      # no GGUF or other copies of the same weights
QUOTE = '<|"|>'               # how Gemma 4 quotes a string inside a tool call


class Dialect:
    """How one family of models marks its thinking and its tool calls in the text it writes. Nothing here is needed to chat: a
    model that matches none of them is simply a plain one, whose whole output is the answer."""

    def __init__(self, name, think_open="", think_close="", label="", call_open="", call_close="", parse=None):
        self.name, self.think_open, self.think_close, self.label = name, think_open, think_close, label
        self.call_open, self.call_close, self.parse = call_open, call_close, parse
        self.markers = tuple(m for m in (think_open, think_close, call_open, call_close) if m)


def _json_call(text):
    """`{"name": ..., "arguments": {...}}` (Qwen, Hermes and most others) -> {"name", "args"}, or None."""
    import json
    try:
        data = json.loads(str(text or "").strip())
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    args = data.get("arguments", data.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return {"name": data["name"], "args": args if isinstance(args, dict) else {}}


DIALECTS = (
    Dialect("gemma4", "<|channel>", "<channel|>", "thought\n", "<|tool_call>", "<tool_call|>", lambda t: parse_call(t)),
    Dialect("think-tags", "<think>", "</think>", "", "<tool_call>", "</tool_call>", _json_call),           # Qwen 3, DeepSeek R1 and the many that copy them
    Dialect("tool-tags", "", "", "", "<tool_call>", "</tool_call>", _json_call),
    Dialect("plain"),
)


def dialect_for(vocab, template=""):
    """The dialect whose markers are in this model's vocabulary (or its chat template); a plain one when none is."""
    have = lambda m: m in vocab or m in str(template)      # noqa: E731
    for d in DIALECTS:
        if d.markers and all(have(m) for m in (d.think_open or d.call_open, d.think_close or d.call_close)):
            return d
    return DIALECTS[-1]


class Splitter:
    """Streamed text (control tokens left in) -> (kind, text) pieces: "thinking", "answer", "call" (a tool call's own text), "call_end".
    `drops`: the model's own end-of-turn and padding tokens, which are not text. `in_thinking`: the prompt already opened the thinking."""

    def __init__(self, dialect=None, drops=(), in_thinking=False):
        self.d = dialect or DIALECTS[-1]
        self.markers = tuple(sorted({*self.d.markers, *[x for x in drops if x]}, key=len, reverse=True))
        self.drops = {x for x in drops if x}
        self.mode, self.buf = ("thinking" if in_thinking else "answer"), ""

    def feed(self, text, final=False):
        self.buf += text
        out, d = [], self.d
        while True:
            if self.mode == "label":
                if self.buf.startswith(d.label):
                    self.buf, self.mode = self.buf[len(d.label):], "thinking"
                elif d.label.startswith(self.buf) and not final:
                    break
                else:
                    self.mode = "thinking"
                continue
            hit = min(((self.buf.find(m), -len(m), m) for m in self.markers if m in self.buf), default=None)
            if hit is None:
                keep = 0 if final else max((n for m in self.markers for n in range(1, len(m)) if self.buf.endswith(m[:n])), default=0)
                body, self.buf = (self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:])
                if body:
                    out.append((self.mode, body))
                break
            at, _, marker = hit
            if at:
                out.append((self.mode, self.buf[:at]))
            self.buf = self.buf[at + len(marker):]
            if marker in self.drops:
                continue
            if marker == d.think_open:
                self.mode = "label" if d.label else "thinking"
            elif marker == d.think_close:
                self.mode = "answer"
            elif marker == d.call_open:
                self.mode = "call"
            elif marker == d.call_close:
                out.append(("call_end", ""))
                self.mode = "answer"
        return out


def parse_call(text):
    """`call:name{key:value,...}` as Gemma 4 writes a tool call -> {"name", "args"}, or None when it is not one."""
    m = re.match(r"\s*call:([\w.\-]+)\s*(\{.*\})\s*$", str(text or ""), re.S)
    if not m:
        return None
    src = m.group(2)
    pos = 0

    def skip():
        nonlocal pos
        while pos < len(src) and src[pos] in " \n\t\r":
            pos += 1

    def value():
        nonlocal pos
        skip()
        if src.startswith(QUOTE, pos):
            end = src.index(QUOTE, pos + len(QUOTE))
            out = src[pos + len(QUOTE):end]
            pos = end + len(QUOTE)
            return out
        if src[pos] == "{":
            pos += 1
            out = {}
            while True:
                skip()
                if src[pos] == "}":
                    pos += 1
                    return out
                key = re.match(r"[\w.\-]+", src[pos:]).group(0)
                pos += len(key)
                skip()
                if src[pos] != ":":
                    raise ValueError("a colon was expected")
                pos += 1
                out[key] = value()
                skip()
                if src[pos] == ",":
                    pos += 1
        if src[pos] == "[":
            pos += 1
            out = []
            while True:
                skip()
                if src[pos] == "]":
                    pos += 1
                    return out
                out.append(value())
                skip()
                if src[pos] == ",":
                    pos += 1
        word = re.match(r"[^,}\]\s]+", src[pos:]).group(0)
        pos += len(word)
        if word in ("true", "false"):
            return word == "true"
        if word in ("null", "None"):
            return None
        try:
            return int(word)
        except ValueError:
            try:
                return float(word)
            except ValueError:
                return word

    try:
        return {"name": m.group(1), "args": value()}
    except (ValueError, IndexError, AttributeError):
        return None


def models_dir():
    """Where downloaded engine models go: the ComfyUI models folder when there is one, else the home folder."""
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "funui_llm")
    except Exception:                                               # noqa: BLE001 -- not inside ComfyUI
        return os.path.join(os.path.expanduser("~"), ".cache", "funui_llm")


def local_dir(source):
    """A folder given as it is; a Hugging Face id ("owner/name") as its folder under the models directory."""
    source = str(source or "").strip()
    return source if os.path.isdir(source) else os.path.join(models_dir(), re.split(r"::|@", source)[0].replace("/", "__"))


def _folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class Engine:
    """One model, loaded on demand, shared by every request (one generation at a time)."""

    def __init__(self):
        self.lock = threading.Lock()           # loading and unloading
        self.turn = threading.Lock()           # one generation at a time
        self.model = self.tok = None
        self.source = ""
        self.phase, self.message = "idle", ""
        self.done_bytes = self.total_bytes = 0
        self.device = ""
        self._snap = None                      # (ids, cache) of the last request, minus its final token
        self.cancel = threading.Event()
        self._worker = None

    # -- loading ------------------------------------------------------------
    def status(self):
        size = _folder_size(local_dir(self.source)) if self.phase == "downloading" and self.source else self.done_bytes
        return {"phase": self.phase, "message": self.message, "source": self.source, "device": self.device,
                "downloaded": size, "total": self.total_bytes}

    def _fetch(self, source):
        """-> (folder, weights file or ""): the given folder, or the Hugging Face repo downloaded (once) into the models directory.
        A repo that has no config.json but a ComfyUI-style single weights file (one fp8 file) comes with its base model's config instead:
        `owner/name` finds the file and the base by itself, `owner/name::folder/file.safetensors@owner/base` names them."""
        if os.path.isdir(source):
            return source, ""
        match = re.fullmatch(r"([\w.\-]+/[\w.\-]+)(?:::([^@]+))?(?:@([\w.\-]+/[\w.\-]+))?", source)
        if not match:
            raise ValueError(f"'{source}' is neither a folder nor a Hugging Face id like owner/name")
        repo, wanted, base = match.groups()
        target = local_dir(repo)
        if os.path.exists(os.path.join(target, ".complete")):
            return self._cached(target)
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        info = HfApi().model_info(repo, files_metadata=True)
        files = {s.rfilename: s.size or 0 for s in info.siblings}
        single = wanted or ("" if "config.json" in files else next((f for f in files if f.endswith(".safetensors")), ""))
        if single not in files and single:
            raise ValueError(f"{repo} has no file {single}")
        self.total_bytes = sum(size for name, size in files.items() if name.endswith((".safetensors", ".json")) and (not single or name == single))
        self.phase, self.message = "downloading", f"downloading {repo}"
        if not single:
            snapshot_download(repo, local_dir=target, allow_patterns=REPO_FILES)
            open(os.path.join(target, ".complete"), "w").close()
            return target, ""
        base = base or next((t.split(":")[-1] for t in info.tags if t.startswith("base_model:quantized:")), "") or next((t.split(":")[-1] for t in info.tags if t.startswith("base_model:") and t.count(":") == 1), "")
        if not base:
            raise ValueError(f"{repo} holds only a weights file, and no base model is named for its config: write {repo}::{single}@owner/base-model")
        os.makedirs(target, exist_ok=True)
        hf_hub_download(repo, single, local_dir=target)
        snapshot_download(base, local_dir=os.path.join(target, "base"), allow_patterns=["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"], ignore_patterns=["*.safetensors.index.json"])
        with open(os.path.join(target, ".complete"), "w") as f:
            f.write(single)
        return self._cached(target)

    @staticmethod
    def _cached(target):
        """(folder, weights file) of a finished download: a weights file written in `.complete` means config and tokenizer are in `base/`."""
        marker = open(os.path.join(target, ".complete")).read().strip()
        return (os.path.join(target, "base"), os.path.join(target, marker)) if marker else (target, "")

    def load(self, source):
        """Make `source` the loaded model (a no-op when it already is). Blocks; raises with the reason."""
        source = str(source or "").strip()
        if not source:
            raise ValueError("no engine model is set")
        with self.lock:
            if self.model is not None and self.source == source:
                return
            self.model = self.tok = self._snap = None
            self.source, self.done_bytes = source, 0
            try:
                folder, weights = self._fetch(source)
                self.phase, self.message = "loading", f"loading {source}"
                import torch
                from transformers import AutoTokenizer
                cuda = torch.cuda.is_available()
                self.tok = AutoTokenizer.from_pretrained(folder)
                if weights:                                          # one scaled-fp8 file: kept fp8 in memory, widened layer by layer
                    from . import fp8
                    self.model = fp8.build(folder, weights, dtype=torch.bfloat16, device="cuda" if cuda else "cpu")
                else:
                    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):      # text-only models first, then the ones that also see pictures
                        try:
                            self.model = getattr(__import__("transformers"), cls).from_pretrained(folder, dtype="auto", device_map="cuda" if cuda else None).eval()
                            break
                        except (ValueError, KeyError, AttributeError) as exc:
                            failure = exc
                    else:
                        raise failure
                self.device = "cuda" if cuda else "cpu"
                self.phase, self.message = "ready", f"{source} on {self.device}"
            except Exception as exc:                                # noqa: BLE001 -- said in the status, raised to the caller
                self.model = self.tok = None
                self.phase, self.message = "error", str(exc)
                raise

    def use(self, model, tokenizer, name="test", device="cpu"):
        """Install a model that was built elsewhere (the tests' tiny one)."""
        self.model, self.tok, self.source, self.device, self._snap = model, tokenizer, name, device, None
        self.phase, self.message = "ready", f"{name} on {device}"

    def load_async(self, source):
        """Start loading in the background; the status says how far it is."""
        if self._worker is not None and self._worker.is_alive():
            return False

        def work():
            try:
                self.load(source)
            except Exception:                                       # noqa: BLE001 -- already in the status
                pass
        self.phase, self.message, self.source = "loading", f"starting {source}", str(source).strip()
        self._worker = threading.Thread(target=work, daemon=True, name="funui-engine-load")
        self._worker.start()
        return True

    def unload(self):
        with self.lock:
            self.model = self.tok = self._snap = None
            self.phase, self.message = "idle", ""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:                                       # noqa: BLE001
                pass

    # -- generating ---------------------------------------------------------
    def stream(self, messages, **options):
        """One request at a time on the model, independent of ComfyUI: a chat sent while a rewrite runs waits its turn, and neither touches the generation's own memory use."""
        with self.turn:
            yield from self._stream(messages, **options)

    def _stream(self, messages, *, thinking=False, tools=None, max_new_tokens=400, temperature=0.7, top_p=0.92, top_k=50, min_p=0.05,
               repetition_penalty=1.0, presence_penalty=0.0, seed=0, greedy=False, cache=True, lookup=5):
        """Yield {"kind": "thinking" | "answer" | "call", "text"} as the model writes, then {"kind": "end", "calls": [...], "tokens", "seconds", "tps", "cached"}.
        `cache`: reuse the start of the last request when this one begins with it (the long instructions are only read once).
        `lookup`: prompt-lookup decoding, up to that many tokens guessed from the text already there and checked in one pass (0 = off);
        the words written are the same as without it. Raises when the model is not loaded or generation fails."""
        if self.model is None:
            raise RuntimeError("the engine model is not loaded")
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        tok, model = self.tok, self.model
        kwargs = {"tools": tools} if tools else {}
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking), **kwargs)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        dialect = dialect_for(tok.get_vocab(), getattr(tok, "chat_template", "") or "")
        drops = [getattr(tok, n, None) for n in ("eos_token", "bos_token", "pad_token")] + (["<turn|>", "<|turn>", "<|tool_response>", "<tool_response|>"] if dialect.name == "gemma4" else [])
        opened = bool(dialect.think_open) and not dialect.label and text.rstrip().endswith(dialect.think_open)       # the prompt itself opened the thinking
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False, timeout=600)
        seen = {}
        cancel = self.cancel
        cancel.clear()

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                stats["steps"] += 1
                return cancel.is_set()

        class Presence(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if presence_penalty:
                    for row, ids in enumerate(input_ids):
                        scores[row, ids.unique()] -= presence_penalty
                return scores

        sample = not greedy and float(temperature) > 0
        gen = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens), do_sample=sample, repetition_penalty=float(repetition_penalty),
                   stopping_criteria=StoppingCriteriaList([Stop()]), logits_processor=LogitsProcessorList([Presence()]))
        if sample:
            gen.update(temperature=max(float(temperature), 0.01), top_p=float(top_p), top_k=int(top_k), min_p=float(min_p))
        torch.manual_seed(int(seed) if seed else random.randrange(1, 2 ** 31))
        failure, stats, started = [], {"tokens": 0, "cached": 0, "steps": 0}, time.time()
        ids = inputs["input_ids"]

        def run(fast):
            """One generation; `fast`: with the prefix cache and prompt lookup."""
            extra, start, cache_obj = {}, 0, None
            if fast and cache and ids.shape[1] > 1:
                from transformers import DynamicCache
                cache_obj = DynamicCache()
                snap = self._snap
                if snap is not None and len(snap[0]) <= ids.shape[1] - 1 and torch.equal(ids[0, :len(snap[0])].cpu(), snap[0]):
                    cache_obj, start = copy.deepcopy(snap[1]), len(snap[0])
                if ids.shape[1] - 1 > start:
                    model(input_ids=ids[:, start:-1], past_key_values=cache_obj, use_cache=True)
                self._snap = (ids[0, :-1].detach().cpu().clone(), copy.deepcopy(cache_obj))
                extra["past_key_values"] = cache_obj
            if fast and lookup:
                extra["prompt_lookup_num_tokens"] = int(lookup)
            stats["cached"] = start
            out = model.generate(**gen, **extra)
            stats["tokens"] = int(out.shape[1] - ids.shape[1])

        def work():
            try:
                with torch.inference_mode():
                    try:
                        run(True)
                    except Exception:                               # noqa: BLE001 -- a model that cannot do the speed-ups still answers, the plain way
                        if not (cache or lookup) or stats["steps"] or self.cancel.is_set():
                            raise
                        self._snap = None
                        run(False)
            except Exception as exc:                                # noqa: BLE001 -- handed to the reader of the stream
                failure.append(exc)
                streamer.end()
        worker = threading.Thread(target=work, daemon=True, name="funui-engine-generate")
        worker.start()
        split, calls, call_text = Splitter(dialect, drops, opened), [], ""
        for chunk in streamer:
            for kind, part in split.feed(chunk):
                if kind == "call":
                    call_text += part
                elif kind == "call_end":
                    parsed = dialect.parse(call_text) if dialect.parse else None
                    call_text = ""
                    if parsed:
                        calls.append(parsed)
                        yield {"kind": "call", "call": parsed}
                else:
                    yield {"kind": kind, "text": part}
        for kind, part in split.feed("", final=True):
            if kind in ("thinking", "answer"):
                yield {"kind": kind, "text": part}
        worker.join()
        if failure:
            raise failure[0]
        took = max(time.time() - started, 1e-6)
        yield {"kind": "end", "calls": calls, "tokens": stats["tokens"], "seconds": round(took, 2), "tps": round(stats["tokens"] / took, 2), "cached": stats["cached"]}


ENGINE = Engine()


class EngineClip:
    """Stands in for a ComfyUI CLIP where the enhancer asks for text: `generate()` hands the request to the engine.
    `on_event` hears each piece as it forms."""

    is_engine = True
    max_context = 0

    MAX_TOOL_ROUNDS = 4

    def __init__(self, engine=None, on_event=None, tools=None, handler=None):
        """`handler(call) -> text` answers the model's tool calls (the answer goes back to the model, which then carries on)."""
        self.engine, self.on_event, self.tools, self.handler, self.calls = engine or ENGINE, on_event, tools, handler, []

    def run(self, system, user, *, rounds=(), thinking=False, max_length=400, temperature=0.7, top_p=0.92, top_k=50, min_p=0.05,
            repetition_penalty=1.0, presence_penalty=0.0, do_sample=True, seed=None, image=None):
        """-> (raw text with the thinking inside <think>..</think>, status); the same as enhance.generate."""
        messages = ([{"role": "system", "content": str(system)}] if str(system).strip() else [])
        for q, a in rounds:
            messages += [{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}]
        messages.append({"role": "user", "content": str(user)})
        started, thought, answer, speed = time.time(), "", "", ""
        self.calls = []
        try:
            for step in range(self.MAX_TOOL_ROUNDS + 1):
                round_calls, said = [], ""
                for ev in self.engine.stream(messages, thinking=thinking, tools=self.tools, max_new_tokens=max_length, temperature=temperature, top_p=top_p,
                                             top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                                             seed=seed or 0, greedy=not do_sample):
                    if ev["kind"] == "thinking":
                        thought += ev["text"]
                    elif ev["kind"] == "answer":
                        answer += ev["text"]
                        said += ev["text"]
                    elif ev["kind"] == "end":
                        round_calls = ev["calls"]
                        speed = f"; {ev['tps']} tokens/s" + (f", {ev['cached']} prompt tokens reused" if ev.get("cached") else "")
                    if self.on_event:
                        self.on_event(ev)
                self.calls += round_calls
                if not (round_calls and self.handler and step < self.MAX_TOOL_ROUNDS):
                    break
                ids = [f"call{step}_{i}" for i in range(len(round_calls))]
                messages.append({"role": "assistant", "content": said.strip(), "tool_calls": [{"id": i, "type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for i, c in zip(ids, round_calls)]})
                for i, c in zip(ids, round_calls):
                    try:
                        result = str(self.handler(c))
                    except Exception as exc:                        # noqa: BLE001 -- the model hears that the tool failed and carries on
                        result = f"Error: {exc}"
                    messages.append({"role": "tool", "tool_call_id": i, "name": c["name"], "content": result})
                    if self.on_event:
                        self.on_event({"kind": "tool_result", "name": c["name"], "result": result})
                answer += "\n\n" if answer.strip() else ""
        except Exception as exc:                                    # noqa: BLE001 -- the enhancer hands the original prompt on and says why
            return "", f"generation failed: {exc}"
        raw = (f"<think>{thought}</think>" if thought.strip() else "") + answer
        note = "; the picture was not sent (the engine reads text only)" if image is not None else ""
        return raw, f"generated {len(answer)} chars in {time.time() - started:.1f}s on {self.engine.device} (engine){speed}{note}"
