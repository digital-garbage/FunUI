"""GGUF models, run by llama.cpp (llama-cpp-python) instead of transformers.

A GGUF file keeps its weights quantized and llama.cpp computes with them as they are, which is much faster than widening them.
The file also carries the model's chat template, which is rendered here with the same renderer transformers uses, so the engine's
dialects, tool calls and thinking work the same as on a transformers model.

llama-cpp-python is fetched on first use (a prebuilt wheel for the machine's graphics card when there is one), so FunUI does not
need it until a .gguf is picked. Imports of it stay inside functions.
"""

import codecs
import os
import subprocess
import sys
import threading
import time

WHEELS = "https://abetlen.github.io/llama-cpp-python/whl/"
CUDA_WHEELS = (("cu125", (12, 5)), ("cu124", (12, 4)), ("cu123", (12, 3)), ("cu122", (12, 2)), ("cu121", (12, 1)))


def wheel_index():
    """The prebuilt-wheel index for this machine: its CUDA version when torch sees a card, else the processor one."""
    try:
        import torch
        if torch.cuda.is_available() and torch.version.cuda:
            have = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
            for name, ver in CUDA_WHEELS:
                if have >= ver:
                    return WHEELS + name, True
    except Exception:                                               # noqa: BLE001 -- no torch or no card: the processor wheel
        pass
    return WHEELS + "cpu", False


def ensure_llama(say=lambda text: None):
    """-> the llama_cpp module, installing it first when it is missing. Raises with the reason when that fails."""
    try:
        import llama_cpp
        return llama_cpp
    except ImportError:
        pass
    index, card = wheel_index()
    say(f"installing llama-cpp-python ({'for the graphics card' if card else 'processor build'}), once")
    attempts = [index] + ([WHEELS + "cpu"] if card else [])
    failure = ""
    for url in attempts:
        done = subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--prefer-binary", "--only-binary=:all:", "--extra-index-url", url],
                              capture_output=True, text=True)
        if done.returncode == 0:
            import importlib
            importlib.invalidate_caches()
            try:
                import llama_cpp
                return llama_cpp
            except ImportError as exc:
                failure = str(exc)
        else:
            failure = (done.stderr or done.stdout).strip().splitlines()[-1:] and (done.stderr or done.stdout).strip().splitlines()[-1] or "pip failed"
    raise RuntimeError(f"llama-cpp-python could not be installed ({failure}). Install it by hand: pip install llama-cpp-python --extra-index-url {index}")


class Backend:
    """A loaded GGUF model. Stands in for both the model and the tokenizer of the engine (`chat_template` and `eos` are read from the file)."""

    gguf = True
    funui_precision = "llama.cpp"

    def __init__(self, llm, path):
        self.llm, self.path = llm, path
        meta = getattr(llm, "metadata", {}) or {}
        self.chat_template = meta.get("tokenizer.chat_template", "") or ""
        self.name = meta.get("general.name", os.path.basename(path))
        self.device = "gguf"
        self.lock = threading.Lock()
        self.projector = ""

    def get_vocab(self) -> dict:
        """The engine asks the tokenizer for its vocabulary to spot a dialect; a GGUF's chat template names the markers just as well."""
        return {}

    def special(self, token) -> str:
        try:
            return self.llm.detokenize([token], special=True).decode("utf-8", "ignore")
        except Exception:                                           # noqa: BLE001 -- a model without that token
            return ""

    @property
    def bos(self) -> str:
        return self.special(self.llm.token_bos()) if self.llm.token_bos() >= 0 else ""

    @property
    def eos(self) -> str:
        return self.special(self.llm.token_eos()) if self.llm.token_eos() >= 0 else ""

    def render(self, messages, tools=None, thinking=False) -> str:
        """The chat template of the file, with the generation prompt at the end."""
        from transformers.utils.chat_template_utils import render_jinja_template
        if not self.chat_template:
            raise ValueError("the .gguf file carries no chat template")
        out = render_jinja_template([messages], tools=tools, chat_template=self.chat_template, add_generation_prompt=True,
                                    bos_token=self.bos, eos_token=self.eos, enable_thinking=bool(thinking))
        out = out[0] if isinstance(out, tuple) else out
        return out[0] if isinstance(out, list) else out

    def text_stream(self, text, *, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, seed, greedy,
                    stops=(), cancel=None, stats=None):
        """Yield the model's text as it forms (control tokens left in, as the engine's splitter expects). Reuses the start of the last prompt."""
        llm = self.llm
        tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        if stats is not None:
            stats["prompt"] = len(tokens)
        end = {llm.token_eos()}
        for marker in stops:
            ids = llm.tokenize(marker.encode("utf-8"), add_bos=False, special=True)
            if len(ids) == 1:
                end.add(ids[0])
        if seed:
            llm.set_seed(int(seed))
        before = getattr(llm, "n_tokens", 0)
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        count = 0
        # a prompt longer than the window cannot be run: say so instead of crashing inside llama.cpp
        window = llm.n_ctx()
        if len(tokens) + 8 > window:
            raise ValueError(f"the request is {len(tokens)} tokens but the model's window is {window}; raise the context or shorten the conversation")
        for token in llm.generate(tokens, top_k=int(top_k), top_p=float(top_p), min_p=float(min_p), temp=0.0 if greedy or float(temperature) <= 0 else float(temperature),
                                  repeat_penalty=float(repetition_penalty), presence_penalty=float(presence_penalty), reset=True):
            if token in end or (cancel is not None and cancel.is_set()):
                break
            count += 1
            if stats is not None:
                stats["steps"] = count
            piece = decoder.decode(llm.detokenize([token], special=True))
            if piece:
                yield piece
            if count >= max_new_tokens or count + len(tokens) >= window - 1:
                break
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail
        if stats is not None:
            stats["tokens"], stats["cached"] = count, 0


def find_projector(path) -> str:
    """The picture projector (`*mmproj*.gguf`) that sits beside the model file, or ""."""
    folder = os.path.dirname(str(path))
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if "mmproj" in name.lower() and name.lower().endswith(".gguf"):
            return os.path.join(folder, name)
    return ""


def projector_handler(llama_cpp, projector, name=""):
    """The llama.cpp handler that reads pictures through `projector`; the one for the model's family when there is one."""
    fmt = llama_cpp.llama_chat_format
    family = "Gemma4ChatHandler" if "gemma-4" in name.lower() or "gemma4" in name.lower() else "MTMDChatHandler"
    return getattr(fmt, family, fmt.MTMDChatHandler)(clip_model_path=projector, verbose=False)


def load(path, *, cuda, context=0, say=lambda text: None):
    """-> a Backend for the .gguf file `path` (the first part of a split model: llama.cpp finds the others beside it).
    A projector file beside it is loaded too, so the model can see pictures."""
    llama_cpp = ensure_llama(say)
    started = time.time()
    projector, handler = find_projector(path), None
    if projector:
        try:
            handler = projector_handler(llama_cpp, projector, os.path.basename(str(path)))
        except Exception as exc:                                    # noqa: BLE001 -- the model still writes text without it
            say(f"engine: the picture projector {os.path.basename(projector)} could not be loaded ({exc}); text only")
            projector = ""
    llm = llama_cpp.Llama(model_path=str(path), n_gpu_layers=-1 if cuda else 0, n_ctx=int(context) or 16384, flash_attn=bool(cuda), verbose=False, chat_handler=handler)
    say(f"engine: llama.cpp loaded {os.path.basename(str(path))}" + (f" with {os.path.basename(projector)}" if projector else "") + f" in {time.time() - started:.0f}s")
    backend = Backend(llm, str(path))
    backend.projector = projector
    return backend
