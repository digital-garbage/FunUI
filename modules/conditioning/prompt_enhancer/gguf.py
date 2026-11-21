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
        self.projector, self.handler = "", None

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

    def marker(self) -> str:
        """The placeholder the projector replaces with the picture's tokens."""
        return self.handler._mtmd_cpp.mtmd_default_marker().decode("utf-8")

    @staticmethod
    def png(image) -> bytes:
        """The first picture of a ComfyUI IMAGE tensor (batch, height, width, channels; 0 to 1), as PNG bytes."""
        import io
        import numpy as np
        from PIL import Image
        frame = image[0] if getattr(image, "ndim", 0) == 4 else image
        array = np.clip(255.0 * frame.cpu().numpy() if hasattr(frame, "cpu") else 255.0 * np.asarray(frame), 0, 255).astype("uint8")
        out = io.BytesIO()
        Image.fromarray(array[..., :3]).save(out, format="PNG")
        return out.getvalue()

    def render(self, messages, tools=None, thinking=False, picture=False) -> str:
        """The chat template of the file, with the generation prompt at the end. `picture`: the last user message opens with the picture's placeholder."""
        if picture:
            messages = [dict(m) for m in messages]
            last = max(i for i, m in enumerate(messages) if m["role"] == "user")
            messages[last]["content"] = f"{self.marker()}\n\n{messages[last]['content']}"
        from transformers.utils.chat_template_utils import render_jinja_template
        if not self.chat_template:
            raise ValueError("the .gguf file carries no chat template")
        out = render_jinja_template([messages], tools=tools, chat_template=self.chat_template, add_generation_prompt=True,
                                    bos_token=self.bos, eos_token=self.eos, enable_thinking=bool(thinking))
        out = out[0] if isinstance(out, tuple) else out
        return out[0] if isinstance(out, list) else out

    def read_picture(self, text, picture) -> list:
        """Run the prompt and the picture through the projector into the model's memory. -> the prompt's tokens (the picture's included)."""
        import ctypes
        llm, handler = self.llm, self.handler
        handler._init_mtmd_context(llm)
        cpp = handler._mtmd_cpp
        bitmap = handler._create_bitmap_from_bytes(picture)
        chunks = cpp.mtmd_input_chunks_init()
        try:
            wrapped = cpp.mtmd_input_text()
            raw = text.encode("utf-8")
            wrapped.text, wrapped.text_len, wrapped.add_special, wrapped.parse_special = raw, len(raw), True, True
            bitmaps = (cpp.mtmd_bitmap_p_ctypes * 1)(bitmap)
            if cpp.mtmd_tokenize(handler.mtmd_ctx, chunks, ctypes.byref(wrapped), bitmaps, 1) != 0:
                raise ValueError("the picture projector could not read the request")
            llm.reset()
            llm._ctx.kv_cache_clear()
            import llama_cpp
            for i in range(cpp.mtmd_input_chunks_size(chunks)):
                chunk = cpp.mtmd_input_chunks_get(chunks, i)
                if chunk is None:
                    continue
                if cpp.mtmd_input_chunk_get_type(chunk) == cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    count = ctypes.c_size_t()
                    pointer = cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(count))
                    if pointer and count.value:
                        llm.eval([pointer[j] for j in range(count.value)])
                else:
                    past = llama_cpp.llama_pos(0)
                    if cpp.mtmd_helper_eval_chunk_single(handler.mtmd_ctx, llm._ctx.ctx, chunk, llama_cpp.llama_pos(llm.n_tokens), llama_cpp.llama_seq_id(0), llm.n_batch, False, ctypes.byref(past)) != 0:
                        raise ValueError("the picture could not be read into the model")
                    llm.n_tokens = past.value
            return llm.input_ids[:llm.n_tokens].tolist()
        finally:
            cpp.mtmd_input_chunks_free(chunks)
            cpp.mtmd_bitmap_free(bitmap)

    def text_stream(self, text, *, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, seed, greedy,
                    stops=(), cancel=None, stats=None, picture=None):
        """Yield the model's text as it forms (control tokens left in, as the engine's splitter expects). Reuses the start of the last prompt."""
        llm = self.llm
        tokens = self.read_picture(text, picture) if picture else llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
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
    backend.projector, backend.handler = projector, handler
    return backend
