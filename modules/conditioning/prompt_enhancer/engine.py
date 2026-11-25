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
import shutil
import threading
import time

from ..._core import log

THINK_BUDGET = 3000      # tokens a thinking model may reason in, on top of the length setting for its answer
REPO_FILES = ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.tiktoken"]      # no GGUF or other copies of the same weights
MAX_PICTURES = 4          # pictures sent to the model in one request

SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})(\.[A-Za-z0-9]+)")


AUX_GGUF = re.compile(r"mmproj|imatrix", re.I)           # files that sit beside a model but are not one: the picture projector, an importance matrix


def main_first(names) -> list:
    """The picked files with a model file first and the helpers (a picture projector) after it; an importance matrix is no use here and is left out."""
    names = [n for n in names if n and not re.search("imatrix", n, re.I)]
    return [n for n in names if not AUX_GGUF.search(n)] + [n for n in names if AUX_GGUF.search(n)]


def reuse_sibling(repo, target, name, size=0) -> bool:
    """Put `name` into `target` from another download of the same repo that already has it (same size), so a file is never fetched twice. -> whether it is there."""
    here = os.path.join(target, name)
    if os.path.isfile(here):
        return True
    root = local_dir(repo)
    folder = os.path.dirname(root) or "."
    for other in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        path = os.path.join(folder, other, name)
        if os.path.join(folder, other) != target and other.startswith(os.path.basename(root) + "__") and os.path.isfile(path) and (not size or os.path.getsize(path) == size):
            os.makedirs(os.path.dirname(here), exist_ok=True)
            try:
                os.link(path, here)
            except OSError:
                shutil.copyfile(path, here)
            return True
    return False


def shard_set(name, files) -> list:
    """The parts of a split model that `name` is one of (model-00002-of-00004.safetensors -> all four, in order), or [] for a file that is not a part.
    Raises when a part is not in the repo."""
    m = SHARD.fullmatch(str(name or ""))
    if not m:
        return []
    prefix, total, ext = m.group(1), int(m.group(3)), m.group(4)
    parts = [f"{prefix}-{i:05d}-of-{total:05d}{ext}" for i in range(1, total + 1)]
    gone = [f for f in parts if f not in files]
    if gone:
        raise ValueError(f"the repo has no part {gone[0]} (a split model needs all {total} parts)")
    return parts


def shard_index(part, files) -> str:
    """The index file of a split safetensors model (the file that says which part holds which weight), or ""."""
    m = SHARD.fullmatch(str(part or ""))
    if not m:
        return ""
    return next((f for f in (f"{m.group(1)}{m.group(4)}.index.json", f"{m.group(1)}.safetensors.index.json") if f in files), "")


BIG_FILE = 64 * 2 ** 20            # from this size a download keeps its half-finished file and continues it next time


def pull_file(repo, filename, target, size=0):
    """`hf_hub_download` into `target`, except that a big file stops where it was cut and continues there on the next try: this
    huggingface_hub throws a half-finished file away. The unfinished bytes sit beside the file as `<name>.part` until it is whole."""
    import huggingface_hub as hub
    if size < BIG_FILE:
        return hub.hf_hub_download(repo, filename, local_dir=target)
    from urllib.parse import urlparse
    from huggingface_hub.file_download import http_get
    url = hub.hf_hub_url(repo, filename)
    meta = hub.get_hf_file_metadata(url)
    final = os.path.join(target, filename)
    if os.path.isfile(final) and (not meta.size or os.path.getsize(final) == meta.size):
        return final                                                 # whole already
    headers = hub.utils.build_hf_headers()
    where = meta.location or url
    if urlparse(where).netloc != urlparse(url).netloc:                # a storage host: the token is not sent along
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    else:
        where = url
    part = final + ".part"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if meta.size and have > meta.size:
        have = 0                                                     # not a piece of this file: start over
    with open(part, "ab" if have else "wb") as out:
        if not meta.size or have < meta.size:
            http_get(where, out, resume_size=have, headers=headers, expected_size=meta.size or None, displayed_filename=filename)
    os.replace(part, final)
    return final


def same_model(a, b):
    """Whether two engine model names are the same model: the same repo, the same files in any order, the same base."""
    a, b = str(a or "").strip(), str(b or "").strip()
    if a == b:
        return True
    parts = [re.fullmatch(r"([\w.\-]+/[\w.\-]+)(?:::([^@]+))?(?:@([\w.\-]+/[\w.\-]+))?", x) for x in (a, b)]
    if not all(parts):
        return False
    key = lambda m: (m.group(1), sorted(f.strip() for f in (m.group(2) or "").split("|") if f.strip()), m.group(3) or "")      # noqa: E731
    return key(parts[0]) == key(parts[1])


def _say(message):
    """A line in FunUI's log: the engine does not run through ComfyUI, so this is how its loading and writing can be followed there."""
    try:
        log.info("FunUI Prompt enhancer", message)
    except Exception:                                               # noqa: BLE001 -- a log line never stops the model
        pass


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
    import re
    from .helper import loose_json
    raw = str(text or "").strip()
    xml = re.match(r"<function=([\w.-]+)>(.*?)(?:</function>|\Z)", raw, re.S)           # Qwen 3.5: <function=name><parameter=key>value</parameter></function>
    if xml:
        got = {}
        for key, value in re.findall(r"<parameter=([\w.-]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|\Z)", xml.group(2), re.S):
            try:
                got[key] = json.loads(value) if value[:1] in "[{" or value in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", value) else value
            except ValueError:
                got[key] = value
        return {"name": xml.group(1), "args": got}
    try:
        data = loose_json(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    args = data.get("arguments", data.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = loose_json(args)
        except ValueError:
            return None
    return {"name": data["name"], "args": args if isinstance(args, dict) else {}}


DIALECTS = (
    Dialect("gemma4", "<|channel>", "<channel|>", "thought\n", "<|tool_call>", "<tool_call|>", lambda t: parse_call(t)),
    Dialect("think-tags", "<think>", "</think>", "", "<tool_call>", "</tool_call>", _json_call),           # Qwen 3, DeepSeek R1 and the many that copy them
    Dialect("tool-tags", "", "", "", "<tool_call>", "</tool_call>", _json_call),
    Dialect("plain"),
)


def close_thinking(text, dialect, thinking) -> str:
    """With thinking off, a Gemma 4 prompt ends with an empty, already closed thought block (as Google's prompt-format guide advises), so the model
    goes straight to its answer instead of reasoning aloud into it. Other dialects and prompts that already end that way are left alone."""
    if thinking or dialect.name != "gemma4":
        return text
    tail = text.rstrip()
    done = dialect.think_open + dialect.label + dialect.think_close
    if tail.endswith(dialect.think_close) or not tail.endswith("<|turn>model"):
        return text
    return tail + "\n" + done


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

    def value(ctx="obj"):
        nonlocal pos
        skip()
        if src[pos] in "\"'":                                  # a text in plain quotes, as models trained on JSON write it
            q, i, buf = src[pos], pos + 1, []
            while i < len(src) and src[i] != q:
                if src[i] == "\\" and i + 1 < len(src):
                    i += 1
                    buf.append({"n": "\n", "t": "\t"}.get(src[i], src[i]))
                else:
                    buf.append(src[i])
                i += 1
            if i >= len(src):
                raise ValueError("a quote was not closed")
            pos = i + 1
            return "".join(buf)
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
                out[key] = value("obj")
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
                out.append(value("list"))
                skip()
                if src[pos] == ",":
                    pos += 1
        # a text written without its quotes may hold spaces, brackets and commas: it runs to the next argument's name or to the closing bracket
        end = re.compile(r"(.*?)(?=\s*,\s*[\w.\-]+\s*:|\s*[}\]]|\s*$)" if ctx == "obj" else r"(.*?)(?=\s*[,\]]|\s*$)", re.S)
        word = end.match(src, pos).group(1)
        if not word:
            raise ValueError("a value was expected")
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


def comfy_file(name):
    """The path of a file in ComfyUI's text encoders list (or a path as given)."""
    if os.path.isfile(name):
        return name
    try:
        import folder_paths
        path = folder_paths.get_full_path("text_encoders", name) or folder_paths.get_full_path("clip", name)
    except Exception:                                               # noqa: BLE001 -- not inside ComfyUI
        path = None
    if not path:
        raise ValueError(f"'{name}' is not in ComfyUI's text encoders list")
    return path


_QUANT_WORDS = re.compile(r"[-_.](fp8|fp16|bf16|fp32|scaled|e4m3fn|e5m2|nvfp4|mxfp8|int8|int4|comfy|split|ggml)(?=[-_.]|$)", re.IGNORECASE)


def guess_base(filename):
    """The Hugging Face model a ComfyUI text encoder file was made from, found by its name: -> "owner/name" of a repo that has a config.json."""
    from huggingface_hub import HfApi
    stem = os.path.splitext(os.path.basename(filename))[0]
    core = _QUANT_WORDS.sub("", stem).strip("-_. ")
    api = HfApi()
    bad = re.compile(r"gguf|mlx|awq|gptq|comfy|fp8|nvfp4|int[48]|exl|onnx", re.IGNORECASE)
    found = [m.id for m in api.list_models(search=core, limit=25) if not bad.search(m.id.split("/")[-1])]
    found = [mid for mid in found if mid.split("/")[-1].lower() == core.lower()]        # only a repo named exactly like the file: a loose match would be somebody's finetune
    for mid in found:
        try:
            if any(f.rfilename == "config.json" for f in api.model_info(mid).siblings):
                return mid
        except Exception:                                           # noqa: BLE001 -- the next one
            continue
    raise ValueError(f"no Hugging Face model named '{core}' was found for its config: name the base model, as comfy:{os.path.basename(filename)}@owner/name")


def write_embedded_tokenizer(weights, folder) -> bool:
    """ComfyUI's text-encoder files carry the tokenizer (`tokenizer_json`): write it as tokenizer.json in `folder`. -> whether one was there."""
    try:
        from safetensors import safe_open
        with safe_open(str(weights), "pt") as f:
            if "tokenizer_json" not in f.keys():
                return False
            data = bytes(f.get_tensor("tokenizer_json").tolist())
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tokenizer.json"), "wb") as out:
            out.write(data)
        return True
    except Exception:                                               # noqa: BLE001 -- the caller says why the tokenizer failed
        return False


def label(source):
    """A model's name for a status line: the repo and, for files of it, the one file or how many."""
    source = str(source or "")
    repo, _, rest = source.partition("::")
    files = [f for f in rest.partition("@")[0].split("|") if f]
    if len(files) == 1:
        return f"{repo}: {files[0]}"
    return f"{repo} ({len(files)} files)" if files else source.partition("@")[0]


def comfy_interrupted() -> bool:
    """Whether ComfyUI was told to interrupt what it is running (its Stop button). False outside ComfyUI."""
    try:
        import comfy.model_management as mm
        return bool(mm.processing_interrupted())
    except Exception:                                               # noqa: BLE001 -- not inside ComfyUI
        return False


ASKING = threading.local()                    # `.stop`: the flag of the Helper request the current thread is serving, if any


class Cancel(threading.Event):
    """A request's stop flag. With `interruptible`, ComfyUI's own interrupt (the Stop button) sets it too: used for a rewrite that runs inside a generation."""

    def __init__(self, interruptible=False, also=None):
        super().__init__()
        self.interruptible = interruptible
        self.also = also                          # another flag that stops it too: the Helper's own Stop, for the request this thread is serving

    def is_set(self):
        return super().is_set() or (self.interruptible and comfy_interrupted()) or (self.also is not None and self.also.is_set())


class Turns:
    """The queue in front of the model: whoever asks first is served first, one at a time (a rewrite, a chat answer, a Helper round, and the loading or
    unloading of the model itself). `take` hands out a place; `wait` blocks until it is that place's turn; `release` gives it up, also when it never got to run."""

    def __init__(self):
        self.cond = threading.Condition()
        self.issued = self.serving = 0
        self.gone = set()
        self.label = ""                                                # what the one being served is doing

    def take(self) -> tuple:
        """-> (a place in the queue, how many are ahead of it, the one being served)."""
        with self.cond:
            place = self.issued
            self.issued += 1
            return place, place - self.serving, self.label

    def wait(self, place, label=""):
        with self.cond:
            while self.serving != place:
                self.cond.wait()
            self.label = label

    def release(self, place):
        with self.cond:
            self.gone.add(place)
            while self.serving in self.gone:
                self.gone.discard(self.serving)
                self.serving += 1
                self.label = ""
            self.cond.notify_all()

    def waiting(self) -> int:
        with self.cond:
            return self.issued - self.serving

    def exclusive(self, label=""):
        """`with turns.exclusive(): ...` waits for its turn like a request does (a model change must not happen under a request that is writing)."""
        import contextlib

        @contextlib.contextmanager
        def held():
            place = self.take()[0]
            try:
                self.wait(place, label)
                yield
            finally:
                self.release(place)
        return held()


class Job:
    """One download or one load in progress: what it is fetching, how far, and what it says."""

    def __init__(self, source):
        self.source, self.dir, self.total, self.message = source, "", 0, ""
        self.lenient = False               # a download takes whatever was picked; only a load needs a model file among them


class Engine:
    """One model, loaded on demand, shared by every request (one generation at a time)."""

    def __init__(self):
        self.lock = threading.Lock()           # loading and unloading
        self.dl_lock, self._dl_worker = threading.Lock(), None          # downloading alone
        self._jobs, self._source_locks = threading.local(), {}
        self.ld = self.dl = None               # the load and the download that are running now (they run side by side)
        self.load_error = ""                   # why the last load failed while an older model kept serving
        self.turns = Turns()                   # one request at a time, first come first served
        self.model = self.tok = None
        self.source = ""                       # what the loaded model came from
        self.activity = {}                     # what a running request is doing: stage, tokens, started
        self.loaded_dir = ""                   # the download folder the loaded model came from ("" for a folder given by hand)
        self.wanted = ""                       # what is being loaded now (it becomes `source` when done)
        self.phase, self.message = "idle", ""
        self.device = ""
        self.precision, self.weights, self.compile_off = "auto", False, False
        self._snap = None                      # (ids, cache) of the last request, minus its final token
        self.cancel = threading.Event()
        self._worker = None

    # -- loading ------------------------------------------------------------
    def status(self):
        """`phase` is the model that serves: idle, loading (only while there is none yet), ready or error. A load or a download that runs beside
        a ready model is in `loading` / `download`, so the old model stays usable and the page can say both."""
        ld, dl = self.ld, self.dl
        shown = self.source
        busy = dict(self.activity)
        if busy:
            busy["seconds"] = round(time.time() - busy.get("started", time.time()), 1)
        message = ld.message if self.phase == "loading" and ld and ld.message else self.message
        fetching = dl or (ld if ld and ld.total else None)
        size = _folder_size(fetching.dir) if fetching and fetching.dir else 0
        return {"busy": busy, "phase": self.phase, "message": message, "source": shown, "label": label(shown), "device": self.device,
                "loading": {"source": ld.source, "label": label(ld.source), "message": ld.message or "starting"} if ld else None,
                "download": {"source": dl.source, "label": label(dl.source), "message": dl.message, "downloaded": size, "total": dl.total} if dl else None,
                "load_error": self.load_error, "downloaded": size, "total": fetching.total if fetching else 0}

    BASE_FILES = ["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"]      # a base model's config and tokenizer, never its weights

    def _job(self):
        """The download or load this thread is running (a scratch one when none): what a fetch reports into."""
        return getattr(self._jobs, "job", None) or Job("")

    def _fetch(self, source):
        """-> (folder, weights file or ""): see `_fetch_files`. The same source is fetched by one thread at a time, so a load asked while a
        download of it runs waits for the files instead of fetching them twice; a different source goes on beside it."""
        with self._source_locks.setdefault(source, threading.Lock()):
            return self._fetch_files(source)

    def _fetch_files(self, source):
        """-> (folder, weights file or ""): where the model is, downloading what is missing once into the models directory.
        * a folder: used as it is;
        * `owner/name`: a Hugging Face repo (all of it, bar copies in other formats);
        * `owner/name::path/file.safetensors`: only that one file of the repo (an fp8 file without the bf16 ones). A repo with no config.json of its own
          takes the config and tokenizer of its base model (read from the repo's tags), or of `@owner/base` when you name one;
        * `comfy:file.safetensors`: a file from ComfyUI's text encoders list. Its config and tokenizer come from the matching Hugging Face
          model, found by the file's name, or from `@owner/base` when you name one."""
        job = self._job()
        if os.path.isdir(source):
            return source, ""
        if os.path.isfile(source) and source.lower().endswith(".gguf"):
            job.dir = ""
            return os.path.dirname(source), source
        from huggingface_hub import HfApi, snapshot_download
        if source.startswith("comfy:"):
            name, _, base = source[6:].partition("@")
            path = comfy_file(name)
            target = os.path.join(models_dir(), "comfy__" + re.sub(r"[^\w.\-]+", "_", os.path.splitext(os.path.basename(path))[0]))
            job.dir = target
            if os.path.exists(os.path.join(target, ".complete")):
                return os.path.join(target, "base"), path
            base = base.strip() or guess_base(os.path.basename(path))
            job.message = f"downloading the config and tokenizer of {base}"
            os.makedirs(target, exist_ok=True)
            snapshot_download(base, local_dir=os.path.join(target, "base"), allow_patterns=self.BASE_FILES, ignore_patterns=["*.safetensors.index.json"])
            open(os.path.join(target, ".complete"), "w").close()
            return os.path.join(target, "base"), path
        match = re.fullmatch(r"([\w.\-]+/[\w.\-]+)(?:::([^@]+))?(?:@([\w.\-]+/[\w.\-]+))?", source)
        if not match:
            raise ValueError(f"'{source}' is neither a folder, a Hugging Face id like owner/name, nor comfy:<file in the text encoders list>")
        repo, wanted, base = match.groups()
        extra = [f.strip() for f in (wanted or "").split("|")[1:] if f.strip()]          # more files of the same repo (the other parts of a split model)
        wanted = (wanted or "").split("|")[0].strip()
        if wanted.lower().endswith(".gguf") or any(f.lower().endswith(".gguf") for f in extra):
            picked = main_first([wanted, *extra])                    # the model first, whatever order the files were picked in; the rest of a split model stay behind it
            if not job.lenient and (not picked or AUX_GGUF.search(picked[0])):
                raise ValueError("pick the model file of the repo as well, not only its picture projector or importance matrix")
            wanted, extra = picked[0], picked[1:]
        target = local_dir(repo) + (("__" + re.sub(r"[^\w.\-]+", "_", os.path.splitext(wanted)[0])) if wanted else "")
        job.dir = target
        if os.path.exists(os.path.join(target, ".complete")) and all(os.path.isfile(os.path.join(target, f)) for f in [wanted, *extra] if f.lower().endswith(".gguf")):
            return self._cached(target)                              # everything picked is there already
        info = HfApi().model_info(repo, files_metadata=True)
        files = {s.rfilename: s.size or 0 for s in info.siblings}
        parts = shard_set(wanted, files)                             # one part of a split model names them all
        if parts:
            wanted, extra = parts[0], [*parts[1:], *[f for f in extra if f not in parts]]
        ggufs = sorted(f for f in files if f.lower().endswith(".gguf"))
        if job.lenient and not wanted and ggufs and "config.json" not in files:
            wanted, extra = ggufs[0], ggufs[1:]                      # nothing picked in a GGUF repo: a download takes all of it
        if not wanted and ggufs and "config.json" not in files:
            raise ValueError(f"{repo} holds GGUF files; pick one with {repo}::{ggufs[0]} (a model split in parts needs only the name of one part). Files: {', '.join(ggufs[:6])}")
        single = wanted or ("" if "config.json" in files else next((f for f in files if f.endswith(".safetensors")), ""))
        for f in [single, *extra]:
            if f and f not in files:
                raise ValueError(f"{repo} has no file {f}")
        pull = [f for f in [single, *extra] if f]
        job.total = sum(size for name, size in files.items() if (name in pull) or (not single and name.endswith((".safetensors", ".json"))) or (single and name.endswith(".json") and name not in pull and not single.lower().endswith(".gguf")))
        job.message = f"downloading {repo}"
        _say(f"engine: downloading {repo}" + (f" ({job.total / 2 ** 30:.1f} GB)" if job.total else ""))
        os.makedirs(target, exist_ok=True)
        if not single:
            snapshot_download(repo, local_dir=target, allow_patterns=REPO_FILES)
            open(os.path.join(target, ".complete"), "w").close()
            return target, ""
        if single.lower().endswith(".gguf"):                         # llama.cpp reads everything from the file itself
            for f in pull:
                if reuse_sibling(repo, target, f, files.get(f, 0)):          # a file another pick of this repo already holds is not fetched again
                    continue
                job.message = f"downloading {f}"
                pull_file(repo, f, target, files.get(f, 0))
            with open(os.path.join(target, ".complete"), "w") as out:
                out.write(single)
            return os.path.join(target, "base"), os.path.join(target, single)
        if not base:
            base = repo if "config.json" in files else (next((t.split(":")[-1] for t in info.tags if t.startswith("base_model:quantized:")), "")
                                                       or next((t.split(":")[-1] for t in info.tags if t.startswith("base_model:") and t.count(":") == 1), ""))
        if not base:
            raise ValueError(f"{repo} holds only a weights file and names no base model for its config: write {repo}::{single}@owner/base-model")
        if len(pull) > 1 and single.lower().endswith(".safetensors"):        # a split model: every part and its index go into one folder, read as a normal checkpoint
            index = shard_index(single, files)
            if not index:
                raise ValueError(f"{repo} has no index file for the parts of {single}")
            for f in [*pull, index]:
                job.message = f"downloading {f}"
                pull_file(repo, f, target, files.get(f, 0))
            if os.path.basename(index) != "model.safetensors.index.json":
                shutil.copyfile(os.path.join(target, index), os.path.join(target, "model.safetensors.index.json"))
            snapshot_download(base, local_dir=target, allow_patterns=self.BASE_FILES, ignore_patterns=["*.safetensors.index.json"])
            open(os.path.join(target, ".complete"), "w").close()
            return target, ""
        pull_file(repo, single, target, files.get(single, 0))
        snapshot_download(base, local_dir=os.path.join(target, "base"), allow_patterns=self.BASE_FILES, ignore_patterns=["*.safetensors.index.json"])
        with open(os.path.join(target, ".complete"), "w") as f:
            f.write(single)
        return self._cached(target)

    @staticmethod
    def _cached(target):
        """(folder, weights file) of a finished download: a weights file written in `.complete` means config and tokenizer are in `base/`."""
        marker = open(os.path.join(target, ".complete")).read().strip()
        return (os.path.join(target, "base"), os.path.join(target, marker)) if marker else (target, "")

    def load(self, source, precision="auto"):
        """Make `source` the loaded model (a no-op when it already is). Blocks; raises with the reason.
        `precision` ("auto", "fp8", "bf16") only matters for a single fp8 file; a different one loads it again.
        The model that is loaded keeps serving while the new one's files are fetched; it is unloaded just before the new one is built, so the two never share memory."""
        source = str(source or "").strip()
        precision = precision if precision in ("auto", "fp8", "bf16") else "auto"
        if not source:
            raise ValueError("no engine model is set")
        if self.model is not None and same_model(self.source, source) and (self.precision == precision or not self.weights):
            return
        return self._load(source, precision)

    def serve(self, source, precision="auto"):
        """What a request calls: use the model that is loaded, whatever the setting names now; only with nothing loaded does it load `source`.
        Switching models is the Load button's job, so a download or a changed pick never swaps the model under a request."""
        if self.model is not None:
            return
        self.load(source, precision)

    def _load(self, source, precision):
        with self.lock:
            if self.model is not None and same_model(self.source, source) and (self.precision == precision or not self.weights):
                return
            job = self.ld = Job(source)
            self._jobs.job = job
            self.load_error = ""
            first = self.model is None                                 # nothing serves yet: the page shows the load as the model's own state
            if first:
                self.source = self.wanted = source
                self.phase = "loading"
            started = time.time()
            job.message = f"looking for {source}"
            _say(f"engine: loading {source} (the first time downloads it)")
            try:
                folder, weights = self._fetch(source)
                weighted = bool(weights)
                self._remember(source, job.dir)                       # what to type to use this download again
                _say(f"engine: files are in place after {time.time() - started:.0f}s, loading the model")
                job.message = f"loading {source}"
                import torch
                from transformers import AutoTokenizer
                cuda = torch.cuda.is_available()
                if self.model is not None:                                # files are here: the old model goes first, so the two never share memory
                    with self.turns.exclusive("unloading the model"):      # a request that is writing finishes on it first
                        self.model = self.tok = self._snap = None
                        self.phase, self.message = "loading", f"loading {source}"
                        self.source = self.wanted = source
                    import gc
                    gc.collect()
                    if cuda:
                        torch.cuda.empty_cache()
                    first = True
                if weights and weights.lower().endswith(".gguf"):    # llama.cpp runs it, quantized as it is
                    from . import gguf
                    model = tok = gguf.load(weights, cuda=cuda, say=lambda t: setattr(job, "message", t) or _say(t))
                    weighted = False
                else:
                    try:
                        tok = AutoTokenizer.from_pretrained(folder)
                    except Exception as exc:                            # noqa: BLE001 -- a base without a tokenizer file of the fast kind: the weights file may carry one
                        if not (weights and write_embedded_tokenizer(weights, folder)):
                            raise ValueError(f"{exc} (the model has no tokenizer.json; sentencepiece must be installed to read its older tokenizer file)") from exc
                        tok = AutoTokenizer.from_pretrained(folder)
                    if weights:                                          # one scaled-fp8 file: kept fp8 in memory, widened layer by layer
                        from . import fp8
                        def step(done, total):
                            job.message = f"loading the weights: {100 * done // max(total, 1)}%"
                        model = fp8.build(folder, weights, dtype=torch.bfloat16, device="cuda" if cuda else "cpu", progress=step, precision=precision)
                    else:
                        for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):      # text-only models first, then the ones that also see pictures
                            try:
                                model = getattr(__import__("transformers"), cls).from_pretrained(folder, dtype="auto", device_map="cuda" if cuda else None).eval()
                                break
                            except (ValueError, KeyError, AttributeError) as exc:
                                failure = exc
                        else:
                            raise failure
                how = getattr(model, "funui_precision", "")
                with self.turns.exclusive("switching the model"):
                    self.model, self.tok, self._snap = model, tok, None
                    self.source, self.precision, self.weights, self.compile_off = source, precision, weighted, False
                    self.loaded_dir = "" if os.path.isdir(source) else job.dir
                    self.device = "cuda" if cuda else "cpu"
                    self.phase, self.message = "ready", f"{label(source)} on {self.device}" + (f", {how}" if how else "")
                _say(f"engine: {label(source)} is ready on {self.device}" + (f" as {how}" if how else "") + f" ({time.time() - started:.0f}s)")
            except Exception as exc:                                # noqa: BLE001 -- said in the status, raised to the caller
                if first:
                    self.model = self.tok = None
                    self.phase, self.message = "error", str(exc)
                else:
                    self.load_error = f"could not load {label(source)}: {exc}"          # the old model is still there and still serves
                _say(f"engine: could not load {source}: {exc}")
                raise
            finally:
                self.ld = None

    def use(self, model, tokenizer, name="test", device="cpu"):
        """Install a model that was built elsewhere (the tests' tiny one)."""
        self.model, self.tok, self.source, self.device, self._snap, self.weights = model, tokenizer, name, device, None, False
        self.phase, self.message = "ready", f"{name} on {device}"

    def _remember(self, source, folder):
        """Write what the downloaded folder was fetched from beside it, so it can be picked again."""
        if not os.path.isdir(source) and folder and os.path.isdir(folder):
            try:
                open(os.path.join(folder, ".source"), "w").write(source)
            except OSError:
                pass

    def download(self, source):
        """Fetch a model's files into the models directory, and nothing else: a model that is loaded stays loaded and untouched.
        Blocks; raises with the reason."""
        source = str(source or "").strip()
        if not source:
            raise ValueError("no engine model is set")
        with self.dl_lock:
            keep = (self.phase, self.message)
            idle = self.model is None
            job = self.dl = Job(source)
            job.lenient = True
            self._jobs.job = job
            job.message = f"downloading {label(source)}"
            try:
                self._fetch(source)
                self._remember(source, job.dir)
                _say(f"engine: downloaded {source}")
                done = f"downloaded {label(source)}"
            except Exception as exc:                                # noqa: BLE001 -- said in the status, raised to the caller
                _say(f"engine: could not download {source}: {exc}")
                failed = f"download failed: {exc}"
                if idle and self.ld is None:
                    self.phase, self.message = "error", failed
                else:
                    self.load_error = failed
                raise
            finally:
                self.dl = None
            if self.ld is None:                                     # a load that runs now owns the message
                self.phase, self.message = ("idle", done) if idle else (keep[0], f"{keep[1]}. {done[0].upper()}{done[1:]}.")

    def download_async(self, source):
        """Start downloading in the background (the status says how far it is). False while another download runs; a load does not stop it."""
        if self._dl_worker is not None and self._dl_worker.is_alive():
            return False

        def work():
            try:
                self.download(source)
            except Exception:                                       # noqa: BLE001 -- already in the status
                pass
        self._dl_worker = threading.Thread(target=work, daemon=True, name="funui-engine-download")
        self._dl_worker.start()
        return True

    def load_async(self, source, precision="auto"):
        """Start loading in the background; the status says how far it is. False while another load runs; a download does not stop it."""
        if self._worker is not None and self._worker.is_alive():
            return False

        def work():
            try:
                self.load(source, precision)
            except Exception:                                       # noqa: BLE001 -- already in the status
                pass
            finally:
                self.ld = None
        self.ld = Job(str(source).strip())
        self.ld.message = f"starting {source}"
        self._worker = threading.Thread(target=work, daemon=True, name="funui-engine-load")
        self._worker.start()
        return True

    @staticmethod
    def _source_of(path, name):
        """What to load to use this download again: what it was fetched from, else the repo its folder name spells."""
        try:
            said = open(os.path.join(path, ".source")).read().strip()
            if said:
                return said
        except OSError:
            pass
        owner, sep, repo = name.partition("__")
        return f"{owner}/{repo}" if sep and "__" not in repo else path

    def downloads(self):
        """The models this engine downloaded: [{"name": folder name, "size", "loaded"}], biggest first."""
        root = models_dir()
        out = []
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                out.append({"name": name, "source": self._source_of(path, name), "size": _folder_size(path), "loaded": bool(self.model is not None and self.loaded_dir == path)})
        return sorted(out, key=lambda m: -m["size"])

    def delete(self, name):
        """Remove one downloaded model from disk (it is unloaded first when it is the one in use). Only a folder directly under the models directory can go."""
        import shutil
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise ValueError("not a downloaded model")
        path = os.path.join(models_dir(), name)
        if not os.path.isdir(path):
            raise ValueError(f"'{name}' is not a downloaded model")
        if self.model is not None and self.loaded_dir == path:
            self.unload()
        with self.lock:
            shutil.rmtree(path)

    def unload(self):
        with self.turns.exclusive("unloading the model"), self.lock:
            self.model = self.tok = self._snap = None
            self.phase, self.message = "idle", ""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:                                       # noqa: BLE001
                pass

    # -- generating ---------------------------------------------------------
    def sees(self) -> bool:
        """Whether the loaded model can be shown a picture (a GGUF model with its projector file)."""
        return bool(getattr(self.model, "gguf", False) and getattr(self.model, "projector", ""))

    def stream(self, messages, **options):
        """One request at a time on the model, independent of ComfyUI: a chat sent while a rewrite runs waits its turn, and neither touches the generation's own memory use."""
        if options.get("thinking") and int(options.get("max_new_tokens") or 0) > 0:                      # the reasoning has its own room: the length setting is for what is written after it
            options["max_new_tokens"] = int(options.get("max_new_tokens") or 400) + THINK_BUDGET
        place, ahead, doing = self.turns.take()
        try:
            if ahead:                                                  # the model is busy: say so instead of looking stuck
                yield {"kind": "status", "text": f"waiting for the model: {ahead} request{'s' if ahead > 1 else ''} ahead" + (f" ({doing})" if doing else "")}
            self.turns.wait(place, str(options.get("label") or "writing"))
            options.pop("label", None)
            yield from self._stream(messages, **options)
        finally:
            self.turns.release(place)

    def _stream(self, messages, *, thinking=False, tools=None, max_new_tokens=400, temperature=0.7, top_p=0.92, top_k=50, min_p=0.05,
               repetition_penalty=1.0, presence_penalty=0.0, seed=0, greedy=False, cache=True, lookup=0, compiled=False, image=None, interruptible=False):
        """Yield {"kind": "thinking" | "answer" | "call", "text"} as the model writes, then {"kind": "end", "calls": [...], "tokens", "seconds", "tps", "cached"}.
        `cache`: reuse the start of the last request when this one begins with it (the long instructions are only read once).
        `lookup`: prompt-lookup decoding, up to that many tokens guessed from the text already there and checked in one pass (0 = off);
        the words written are the same as without it. `compiled`: a fixed-size cache and a compiled decode step (faster per word after a first slow request;
        it replaces the prefix cache and the lookup, and a model that cannot do it answers the plain way). Raises when the model is not loaded or generation fails."""
        if self.model is None:
            raise RuntimeError("the engine model is not loaded")
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        tok, model = self.tok, self.model
        if getattr(model, "gguf", False):
            yield from self._stream_gguf(messages, thinking=thinking, tools=tools, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                                         min_p=min_p, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, seed=seed, greedy=greedy, image=image, interruptible=interruptible)
            return
        kwargs = {"tools": tools} if tools else {}
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking), **kwargs)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        dialect = dialect_for(tok.get_vocab(), getattr(tok, "chat_template", "") or "")
        text = close_thinking(text, dialect, thinking)
        drops = [getattr(tok, n, None) for n in ("eos_token", "bos_token", "pad_token")] + (["<turn|>", "<|turn>", "<|tool_response>", "<tool_response|>"] if dialect.name == "gemma4" else [])
        opened = bool(dialect.think_open) and not dialect.label and text.rstrip().endswith(dialect.think_open)       # the prompt itself opened the thinking
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        size = int(inputs["input_ids"].shape[1])
        self.activity = {"stage": "reading the prompt", "tokens": 0, "prompt": size, "started": time.time()}
        yield {"kind": "status", "text": f"reading the prompt ({size} tokens)"}
        me = self
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False, timeout=600)
        seen = {}
        cancel = self.cancel = Cancel(interruptible, getattr(ASKING, "stop", None))         # this request's own stop flag, made once it has the turn

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                stats["steps"] += 1
                me.activity.update(stage="writing", tokens=stats["steps"])
                return cancel.is_set()

        class Presence(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if presence_penalty:
                    for row, ids in enumerate(input_ids):
                        scores[row, ids.unique()] -= presence_penalty
                return scores

        sample = not greedy and float(temperature) > 0
        gen = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens) or max(256, min(1 << 16, int(getattr(model.config, "max_position_embeddings", 0) or 32768) - size)), do_sample=sample, repetition_penalty=float(repetition_penalty),
                   stopping_criteria=StoppingCriteriaList([Stop()]), logits_processor=LogitsProcessorList([Presence()]))
        if sample:
            gen.update(temperature=max(float(temperature), 0.01), top_p=float(top_p), top_k=int(top_k), min_p=float(min_p))
        torch.manual_seed(int(seed) if seed else random.randrange(1, 2 ** 31))
        failure, stats, started = [], {"tokens": 0, "cached": 0, "steps": 0}, time.time()
        ids = inputs["input_ids"]

        def run(fast):
            """One generation; `fast`: with the prefix cache and prompt lookup."""
            extra, start, cache_obj = {}, 0, None
            if fast and compiled and not self.compile_off and str(getattr(model, "device", "")).startswith("cuda"):
                extra["cache_implementation"] = "static"             # transformers compiles the decode step itself for a static cache on a graphics card
                try:                                                 # without CUDA graphs: ComfyUI's memory allocator does not support them
                    from transformers.generation.configuration_utils import CompileConfig
                    extra["compile_config"] = CompileConfig(mode="default", fullgraph=False)
                except Exception:                                    # noqa: BLE001 -- an older library: its own choice
                    pass
            elif fast and cache and ids.shape[1] > 1:
                from transformers import DynamicCache
                cache_obj = DynamicCache()
                snap = self._snap
                if snap is not None and len(snap[0]) <= ids.shape[1] - 1 and torch.equal(ids[0, :len(snap[0])].cpu(), snap[0]):
                    cache_obj, start = copy.deepcopy(snap[1]), len(snap[0])
                if ids.shape[1] - 1 > start:
                    model(input_ids=ids[:, start:-1], past_key_values=cache_obj, use_cache=True)
                self._snap = (ids[0, :-1].detach().cpu().clone(), copy.deepcopy(cache_obj))
                extra["past_key_values"] = cache_obj
            if fast and lookup and not (compiled and not self.compile_off):
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
                        if compiled and not self.compile_off and stats["steps"] <= 2 and not self.cancel.is_set():
                            self.compile_off = True                 # this model or card cannot: off until the model is loaded again
                            _say("engine: the compiled writing step did not work here, writing the plain way from now on")
                            self._snap = None
                            run(False)
                            return
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
        try:
            yield from self._pieces(streamer, split, calls, dialect)
        finally:
            self.activity = {}
            if worker.is_alive():                           # the reader went away (page closed, request dropped): stop writing before the next request starts
                cancel.set()
                worker.join(timeout=60)
        for kind, part in split.feed("", final=True):
            if kind in ("thinking", "answer"):
                yield {"kind": kind, "text": part}
        worker.join()
        if failure:
            raise failure[0]
        took = max(time.time() - started, 1e-6)
        _say(f"engine: wrote {stats['tokens']} tokens in {took:.1f}s ({stats['tokens'] / took:.1f} tokens/s" + (f", {stats['cached']} prompt tokens reused)" if stats["cached"] else ")"))
        yield {"kind": "end", "calls": calls, "tokens": stats["tokens"], "seconds": round(took, 2), "tps": round(stats["tokens"] / took, 2), "cached": stats["cached"]}

    def _stream_gguf(self, messages, *, thinking, tools, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, seed, greedy, image=None, interruptible=False):
        """The same events as `_stream`, from a GGUF model run by llama.cpp (which reuses the unchanged start of the prompt by itself)."""
        model = self.model
        picture = model.pngs(image)[:MAX_PICTURES] if image is not None and model.projector else None
        text = model.render(messages, tools=tools, thinking=thinking, picture=len(picture) if picture else False)
        dialect = dialect_for({}, model.chat_template)
        text = close_thinking(text, dialect, thinking)
        ends = ["<turn|>", "<|turn>", "<|im_end|>", "<end_of_turn>", "<|tool_response>", "<tool_response|>"] if dialect.name == "gemma4" else ["<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
        drops = [model.bos, model.eos, *ends]
        opened = bool(dialect.think_open) and not dialect.label and text.rstrip().endswith(dialect.think_open)
        self.activity = {"stage": "reading the prompt", "tokens": 0, "prompt": 0, "started": time.time()}
        yield {"kind": "status", "text": "reading the prompt"}
        stats, calls, started = {"tokens": 0, "cached": 0, "steps": 0, "prompt": 0}, [], time.time()
        cancel = self.cancel = Cancel(interruptible, getattr(ASKING, "stop", None))
        me = self

        budget = int(getattr(ASKING, "budget", 0) or 0) if thinking and dialect.think_close else 0

        def run(prompt, room, into):
            return model.text_stream(prompt, max_new_tokens=room, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                                     repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, seed=seed, greedy=greedy,
                                     stops=ends, cancel=cancel, stats=into, picture=picture)

        def chunks():
            written, spent, over = "", 0, False
            stream = run(text, int(max_new_tokens), stats)
            for piece in stream:
                me.activity.update(stage="writing", tokens=stats["steps"], prompt=stats["prompt"])
                written += piece
                yield piece
                if budget and split.mode == "thinking":
                    spent += 1
                    if spent >= budget:                      # it has thought long enough: the thought is closed for it and it answers
                        over = True
                        break
            if not over:
                return
            stream.close()
            first = stats["steps"]
            yield dialect.think_close + ("" if dialect.name == "gemma4" else "\n\n")
            more = {"prompt": stats["prompt"]}
            room = int(max_new_tokens) - first if int(max_new_tokens) else 0
            if int(max_new_tokens) and room <= 0:
                stats["tokens"] = first
                return
            for piece in run(text + written + dialect.think_close + ("" if dialect.name == "gemma4" else "\n\n"), room, more):
                me.activity.update(stage="writing", tokens=first + more.get("steps", 0), prompt=stats["prompt"])
                yield piece
            stats["steps"] = stats["tokens"] = first + more.get("steps", 0)
            stats["last"] = more.get("last", stats.get("last", 0))
            if more.get("why"):
                stats["why"] = more["why"]
        split = Splitter(dialect, drops, opened)
        try:
            yield from self._pieces(chunks(), split, calls, dialect)
        finally:
            self.activity = {}
            cancel.set()
        for kind, part in split.feed("", final=True):
            if kind in ("thinking", "answer"):
                yield {"kind": kind, "text": part}
        took = max(time.time() - started, 1e-6)
        # The speed is the writing alone: the time to read a long prompt (a cleared memory after a change of prompt, a picture) is not the model's speed.
        read = max(stats.get("first", started) - started, 0.0)
        writing = stats.get("last", 0) - stats.get("first", 0)
        tps = (stats["tokens"] - 1) / writing if stats["tokens"] > 1 and writing > 0 else stats["tokens"] / took
        _say(f"engine: read {stats['prompt']} prompt tokens in {read:.1f}s, wrote {stats['tokens']} tokens at {tps:.1f} tokens/s (llama.cpp)" + (f"; {stats['why']}" if stats.get("why") and not stats["tokens"] else ""))
        yield {"kind": "end", "calls": calls, "tokens": stats["tokens"], "seconds": round(took, 2), "tps": round(tps, 2), "cached": 0, "read_seconds": round(read, 2), "why": stats.get("why", "")}

    @staticmethod
    def _pieces(streamer, split, calls, dialect):
        """The events the model's text makes as it arrives (tool calls are collected in `calls`)."""
        call_text = ""
        for chunk in streamer:
            for kind, part in split.feed(chunk):
                if kind == "call":
                    call_text += part
                elif kind == "call_end":
                    parsed = dialect.parse(call_text) if dialect.parse else None
                    if not parsed and call_text.strip():        # not read: the model is told so (as the call's own answer) instead of the call vanishing
                        named = re.search(r"([\w.\-]+)", call_text.replace("call:", " ", 1))
                        parsed = {"name": named.group(1) if named else "unknown", "args": {}, "unreadable": call_text.strip()[:300]}
                    call_text = ""
                    if parsed:
                        calls.append(parsed)
                        yield {"kind": "call", "call": parsed}
                else:
                    yield {"kind": kind, "text": part}


ENGINE = Engine()


class Live:
    """What the engine is writing right now, for the Enhance and Chat tabs to show while a rewrite runs inside a generation (the page cannot hear it
    any other way: that run belongs to ComfyUI's queue). Every run keeps its own record, so a chat asked while a rewrite is still writing cannot
    take over or end the rewrite's view; the tabs read the latest rewrite, which stays until the next one starts."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.runs = {}

    def start(self, kind, label=""):
        """-> the run's id, which feed and finish are given so that each event lands in its own run."""
        with self.lock:
            self.seq += 1
            self.runs[self.seq] = {"id": self.seq, "kind": kind, "label": label, "thinking": "", "answer": "", "status": "", "done": False, "tps": 0.0}
            for old in sorted(self.runs)[:-8]:
                self.runs.pop(old)
            return self.seq

    def feed(self, ev, rid=None):
        with self.lock:
            run = self.runs.get(rid if rid is not None else self.seq)
            if not run or run["done"]:
                return
            kind = ev.get("kind")
            if kind in ("thinking", "answer"):
                run[kind] += ev["text"]
            elif kind == "status":
                run["status"] = ev["text"]
            elif kind == "end":
                run.update(done=True, tps=ev.get("tps", 0.0), status="")

    def finish(self, status="", rid=None):
        with self.lock:
            run = self.runs.get(rid if rid is not None else self.seq)
            if run and not run["done"]:
                run.update(done=True, status=status)

    def read(self):
        """The latest rewrite (what the tabs show); {} before any."""
        with self.lock:
            rewrites = [r for r in self.runs.values() if r["kind"] == "rewrite"]
            return dict(max(rewrites, key=lambda r: r["id"])) if rewrites else {}


LIVE = Live()


class EngineClip:
    """Stands in for a ComfyUI CLIP where the enhancer asks for text: `generate()` hands the request to the engine.
    `on_event` hears each piece as it forms."""

    is_engine = True
    max_context = 0

    MAX_TOOL_ROUNDS = 4

    def __init__(self, engine=None, on_event=None, tools=None, handler=None, live="", cache=True, lookup=0, compiled=False):
        """`handler(call) -> text` answers the model's tool calls (the answer goes back to the model, which then carries on).
        `live`: a name for what is being written ("rewrite"): the pieces are then also kept in LIVE for the tabs to show."""
        self.engine, self.on_event, self.tools, self.handler, self.calls, self.live = engine or ENGINE, on_event, tools, handler, [], live
        self.cache, self.lookup, self.compiled = bool(cache), max(0, int(lookup)), bool(compiled)

    def run(self, system, user, *, rounds=(), thinking=False, max_length=0, temperature=0.7, top_p=0.92, top_k=50, min_p=0.05,
            repetition_penalty=1.0, presence_penalty=0.0, do_sample=True, seed=None, image=None):
        """-> (raw text with the thinking inside <think>..</think>, status); the same as enhance.generate."""
        messages = ([{"role": "system", "content": str(system)}] if str(system).strip() else [])
        for q, a in rounds:
            messages += [{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}]
        messages.append({"role": "user", "content": str(user)})
        started, thought, answer, speed = time.time(), "", "", ""
        self.calls = []
        if self.live:
            rid = LIVE.start(self.live, str(user)[:80])
        try:
            empties, loops, think = 0, 0, thinking
            for step in range(self.MAX_TOOL_ROUNDS + 1):
                round_calls, said, why = [], "", ""
                before = (thought, answer)
                for ev in self.engine.stream(messages, thinking=think, tools=self.tools, max_new_tokens=max_length, temperature=temperature, top_p=top_p,
                                             top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                                             seed=seed or 0, greedy=not do_sample, cache=self.cache, lookup=self.lookup, compiled=self.compiled, image=image, interruptible=self.live == "rewrite", label="a rewrite" if self.live == "rewrite" else "a chat answer"):
                    if ev["kind"] == "thinking":
                        thought += ev["text"]
                    elif ev["kind"] == "answer":
                        answer += ev["text"]
                        said += ev["text"]
                    elif ev["kind"] == "end":
                        round_calls, why = ev["calls"], ev.get("why", "")
                        speed = f"; {ev['tps']} tokens/s" + (f", {ev['cached']} prompt tokens reused" if ev.get("cached") else "")
                    if self.live and not (ev["kind"] == "end" and ev.get("calls") and self.handler):          # a round that ends in a tool call is not the end
                        LIVE.feed(ev, rid)
                    if self.on_event:
                        self.on_event(ev)
                self.calls += round_calls
                if not round_calls and not said.strip() and why.startswith("the model ended its turn at once") and empties < 2 and step < self.MAX_TOOL_ROUNDS:
                    empties += 1                                        # an empty answer (the model's first token was its end-of-turn): asked again with another seed
                    seed = (seed or 0) + 7919
                    continue
                if why.startswith("it repeated the same text") and loops < 2 and step < self.MAX_TOOL_ROUNDS:
                    loops += 1                                          # the model got stuck repeating itself: the round is thrown away and asked again, told so, with thinking off
                    thought, answer = before
                    seed, think = (seed or 0) + 104729, False
                    from .helper import app_note
                    last = dict(messages[-1])
                    last["content"] = f"{last.get('content') or ''}\n\n" + app_note("Your last attempt got stuck repeating itself, so it was thrown away. Do not deliberate: decide, then call the tool (or answer) at once.")
                    messages[-1] = last
                    continue
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
            if self.live:
                LIVE.finish(f"failed: {exc}", rid)
            return "", f"generation failed: {exc}"
        if self.live:
            LIVE.finish(rid=rid)
        raw = (f"<think>{thought}</think>" if thought.strip() else "") + answer
        note = "; the picture was not sent (this model reads text only)" if image is not None and not self.engine.sees() else ""
        return raw, f"generated {len(answer)} chars in {time.time() - started:.1f}s on {self.engine.device} (engine){speed}{note}"
