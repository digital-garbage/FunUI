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


CUDA_LIBS = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")


def preload_cuda() -> list:
    """Load the CUDA 12 runtime libraries that sit in the Python environment (torch's and the nvidia-* packages') so the graphics-card build of
    llama.cpp finds them even when the system's library path does not. -> the files loaded."""
    import ctypes
    import glob
    import site
    roots = [p for p in (*site.getsitepackages(), site.getusersitepackages(), *sys.path) if p and os.path.isdir(p)]
    loaded = []
    for lib in CUDA_LIBS:
        for root in dict.fromkeys(roots):
            hits = glob.glob(os.path.join(root, "nvidia", "*", "lib", lib)) + glob.glob(os.path.join(root, "torch", "lib", lib))
            if hits:
                try:
                    ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
                    loaded.append(hits[0])
                    break
                except OSError:
                    continue
    if os.name == "nt":                                             # Windows looks in added folders instead
        for root in roots:
            for folder in glob.glob(os.path.join(root, "nvidia", "*", "bin")) + [os.path.join(root, "torch", "lib")]:
                if os.path.isdir(folder):
                    os.add_dll_directory(folder)
    return loaded


def _import():
    """Import llama_cpp as it is; only when its libraries do not load, put the CUDA libraries of this environment in front and try once more.
    (Loading them first would make torch use those copies too, which is not ours to change.)"""
    try:
        import llama_cpp
        return llama_cpp
    except (ImportError, RuntimeError, OSError):
        if not preload_cuda():
            raise
        import llama_cpp
        return llama_cpp


def _pip(args, say, what):
    """Run pip with its lines going to `say`, so the install does not look stuck. -> (success, last line)."""
    proc = subprocess.Popen([sys.executable, "-m", "pip", "install", *args, "--prefer-binary", "--progress-bar", "off"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in proc.stdout:
        line = line.strip()
        if line:
            last = line
            say(f"{what}: {line[:110]}")
    return proc.wait() == 0, last


def _marker() -> str:
    from . import engine
    return os.path.join(engine.models_dir(), ".llama_installed")


def ensure_llama(say=lambda text: None):
    """-> the llama_cpp module. What it needs is installed here, once (llama-cpp-python, and for a graphics card the CUDA 12 runtime packages);
    when the fresh install does not load in this running process, the error asks for a restart, and the next load tries again.
    Only a build that still fails after that restart is swapped for the processor one. Raises with the reason when nothing works."""
    import importlib
    import importlib.util
    problem, index, card = "", *wheel_index()
    if importlib.util.find_spec("llama_cpp") is not None:
        try:
            return _import()
        except (ImportError, RuntimeError, OSError) as exc:        # installed, but its libraries do not load
            problem = str(exc)
    fresh = not problem
    restarted = os.path.exists(_marker())
    if fresh:
        say("GGUF models need llama-cpp-python, which is not installed yet: installing it now (a restart of ComfyUI may be needed afterwards)")
        for url in [index] + ([WHEELS + "cpu"] if card else []):
            done, last = _pip(["llama-cpp-python", "--only-binary=:all:", "--extra-index-url", url], say, "installing llama-cpp-python")
            if done:
                break
            problem = last or "pip failed"
        else:
            raise RuntimeError(f"llama-cpp-python could not be installed ({problem[:300]}). Install it by hand: pip install llama-cpp-python --extra-index-url {index}")
    if card and ("libcud" in problem or "libcublas" in problem or fresh):
        say("the graphics-card build needs the CUDA 12 runtime: installing it")
        _pip(["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"], say, "installing the CUDA runtime")
    importlib.invalidate_caches()
    try:
        return _import()
    except (ImportError, RuntimeError, OSError) as exc:
        problem = str(exc)
    if not restarted:                                               # a running process often cannot load what it just installed: ask for a restart first
        try:
            os.makedirs(os.path.dirname(_marker()), exist_ok=True)
            open(_marker(), "w").close()
        except OSError:
            pass
        raise RuntimeError("llama.cpp was installed but this running ComfyUI cannot load it yet. Restart ComfyUI, then load the model again.")
    if card:                                                        # still failing after a restart: the processor build, slower but it runs
        say("the graphics-card build of llama.cpp does not load here: installing the processor build instead")
        _pip(["llama-cpp-python", "--force-reinstall", "--no-deps", "--only-binary=:all:", "--extra-index-url", WHEELS + "cpu"], say, "installing llama-cpp-python")
        importlib.invalidate_caches()
        for name in [m for m in sys.modules if m.startswith("llama_cpp")]:
            del sys.modules[name]
        try:
            return _import()
        except (ImportError, RuntimeError, OSError) as exc:
            problem = str(exc)
    raise RuntimeError(f"llama-cpp-python could not be loaded ({problem[:300]}). Install it by hand: pip install llama-cpp-python --extra-index-url {index}")


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

    @classmethod
    def pngs(cls, image) -> list:
        """One PNG per picture: the first frame of each ComfyUI IMAGE in a list, or of one IMAGE."""
        return [cls.png(i) for i in (image if isinstance(image, (list, tuple)) else [image]) if i is not None]

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
        """The chat template of the file, with the generation prompt at the end. `picture`: the last user message opens with a placeholder for each picture (True is one, or the number of pictures)."""
        if picture:
            messages = [dict(m) for m in messages]
            last = max(i for i, m in enumerate(messages) if m["role"] == "user")
            messages[last]["content"] = f"{(self.marker() + chr(10)) * max(1, int(picture))}\n{messages[last]['content']}"
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
        pictures = [picture] if isinstance(picture, (bytes, bytearray)) else list(picture)
        bitmaps = [handler._create_bitmap_from_bytes(p) for p in pictures]
        chunks = cpp.mtmd_input_chunks_init()
        try:
            wrapped = cpp.mtmd_input_text()
            raw = text.encode("utf-8")
            wrapped.text, wrapped.text_len, wrapped.add_special, wrapped.parse_special = raw, len(raw), True, True
            handles = (cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
            if cpp.mtmd_tokenize(handler.mtmd_ctx, chunks, ctypes.byref(wrapped), handles, len(bitmaps)) != 0:
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
            for bitmap in bitmaps:
                cpp.mtmd_bitmap_free(bitmap)

    def can_trim(self) -> bool:
        """Whether the model's memory can drop its newest part (a plain transformer can; a recurrent or hybrid one such as Qwen3.5 cannot)."""
        import llama_cpp
        arch = str((getattr(self.llm, "metadata", None) or {}).get("general.architecture", "")).lower()
        if any(k in arch for k in ("qwen35", "qwen3next", "qwen3_5", "mamba", "rwkv", "jamba", "lfm2", "granitehybrid", "falcon-h1", "nemotron_h", "plamo2", "minimax")):
            return False
        if any(k.endswith(".attention.sliding_window") and str(v) not in ("", "0") for k, v in (getattr(self.llm, "metadata", None) or {}).items()):
            return False                                            # a sliding-window model keeps only the newest window per layer, so it starts from empty instead of trimming
        try:
            return bool(llama_cpp.llama_memory_can_shift(llama_cpp.llama_get_memory(self.llm._ctx.ctx)))
        except Exception:                                           # noqa: BLE001
            return True

    TURN_OPENERS = ("<|im_start|>", "<start_of_turn>", "<|turn>", "<|start_header_id|>")
    KEPT_SNAPSHOTS = 3

    def _snapshot(self, ids):
        """Keep the model's memory after `ids` (just the used part of it: a copy of the whole window would be far too big)."""
        import ctypes
        import llama_cpp
        llm = self.llm
        ctx = llm._ctx.ctx
        size = int(llama_cpp.llama_state_get_size(ctx))
        buf = (ctypes.c_uint8 * size)()
        used = int(llama_cpp.llama_state_get_data(ctx, buf, size))
        snaps = self.__dict__.setdefault("snapshots", [])
        snaps[:] = [s for s in snaps if s[0] != tuple(ids)]
        snaps.append((tuple(ids), bytes(buf[:used])))
        del snaps[:-self.KEPT_SNAPSHOTS]

    def _restore(self, ids, data):
        import ctypes
        import llama_cpp
        llm = self.llm
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        if int(llama_cpp.llama_state_set_data(llm._ctx.ctx, buf, len(data))) != len(data):
            raise RuntimeError("could not restore the saved memory")
        llm.input_ids[:len(ids)] = ids
        llm.n_tokens = len(ids)
        llm._requires_eval = True

    def _resume(self, text, tokens):
        """-> (the tokens still to read, whether to start from empty) for a model whose memory cannot drop its newest part (recurrent, hybrid, sliding window).
        Such a model cannot simply reuse the start of the last prompt, because the answer it wrote now sits at the end of its memory; so a copy of the memory
        is kept at the start of each prompt's last turn, which the next prompt of the same conversation begins with."""
        llm = self.llm
        cached = llm.input_ids[:llm.n_tokens].tolist() if llm.n_tokens else []
        live = len(cached) if cached and cached == tokens[:len(cached)] and len(cached) < len(tokens) else 0
        best = max((s for s in self.__dict__.get("snapshots", []) if len(s[0]) < len(tokens) and list(s[0]) == tokens[:len(s[0])]), key=lambda s: len(s[0]), default=None)
        if best is not None and len(best[0]) > live:
            self._restore(best[0], best[1])
            position = len(best[0])
        elif live:
            position = live
        else:
            llm.reset()
            llm._ctx.kv_cache_clear()
            position = 0
        mark = max((text.rfind(m) for m in self.TURN_OPENERS), default=-1)
        if mark > 0:
            ahead = llm.tokenize(text[:mark].encode("utf-8"), add_bos=False, special=True)
            if position < len(ahead) < len(tokens) and tokens[:len(ahead)] == ahead:
                if not position and llm.n_tokens:
                    llm.reset()
                llm.eval(tokens[position:len(ahead)])
                self._snapshot(ahead)
                position = len(ahead)
        return (tokens[position:], False) if position else (tokens, True)

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
        if len(tokens) + 64 > window:
            raise ValueError(f"the request is {len(tokens)} tokens but the model's window is {window}; shorten the conversation (Summarise long conversations) or the prompts")
        max_new_tokens = min(int(max_new_tokens) or window, window - len(tokens) - 8)           # what is left of the window after the request
        sampling = dict(top_k=int(top_k), top_p=float(top_p), min_p=float(min_p), temp=0.0 if greedy or float(temperature) <= 0 else float(temperature),
                        repeat_penalty=float(repetition_penalty), presence_penalty=float(presence_penalty))
        feed, fresh = tokens, True
        if not picture and not self.can_trim():
            feed, fresh = self._resume(text, tokens)
        sent = False
        for attempt in (0, 1):
            try:
                for token in llm.generate(feed, reset=fresh, **sampling):
                    if token in end or (cancel is not None and cancel.is_set()):
                        break
                    count += 1
                    if stats is not None:
                        stats["steps"] = count
                        stats["last"] = time.time()
                        if count == 1:
                            stats["first"] = stats["last"]           # the prompt has been read by now: speed is counted from here
                    piece = decoder.decode(llm.detokenize([token], special=True))
                    if piece:
                        sent = True
                        yield piece
                    if count >= max_new_tokens or count + len(tokens) >= window - 1:
                        break
                break
            except RuntimeError as exc:                             # llama.cpp ran out of cache slots (reusing the last request's memory): start from empty once
                if attempt or sent or "llama_decode" not in str(exc):
                    if "llama_decode" in str(exc):
                        raise RuntimeError(f"{exc} (a request of {len(tokens)} tokens, the model's window is {window}; the memory is cleared and it still does not fit: raise the context window or shorten the conversation)") from exc
                    raise
                llm.reset()
                llm._ctx.kv_cache_clear()
                tokens = self.read_picture(text, picture) if picture else tokens
                feed, fresh = tokens, True
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


_KEEP = []                                                          # the callback must stay referenced, or ctypes frees it under llama.cpp


def watch_native(llama_cpp, say):
    """Keep what llama.cpp says natively (it is silent otherwise) in a small file, and a marker for as long as this process lives. A marker left behind by
    a process that is gone means it ended abruptly (a crash inside llama.cpp or the graphics driver): the next load reports the last lines it wrote."""
    from . import engine
    folder = engine.models_dir()
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        return
    tail, marker = os.path.join(folder, "llama_native.log"), os.path.join(folder, ".llama_running")
    try:
        old = int(open(marker).read().strip() or 0)
    except (OSError, ValueError):
        old = 0
    if old and old != os.getpid():
        try:
            os.kill(old, 0)
            alive = True
        except OSError:
            alive = False
        if not alive:
            try:
                lines = [x.strip() for x in open(tail, errors="replace").read().splitlines() if x.strip()][-6:]
            except OSError:
                lines = []
            say("engine: the last ComfyUI session ended abruptly while llama.cpp was loaded" + (("; its last lines: " + " | ".join(x[:160] for x in lines)) if lines else ""))
    if not _KEEP:
        try:
            with open(marker, "w") as f:
                f.write(str(os.getpid()))
            import atexit
            atexit.register(lambda: os.path.exists(marker) and os.remove(marker))
            if os.path.exists(tail) and os.path.getsize(tail) > 200_000:
                open(tail, "w").close()
            out = open(tail, "a", buffering=1)

            def log(level, text, _user):
                try:
                    out.write((text or b"").decode("utf-8", "replace"))
                except (OSError, ValueError):
                    pass
            callback = llama_cpp.llama_log_callback(log)
            llama_cpp.llama_log_set(callback, None)
            _KEEP.append(callback)
        except Exception:                                           # noqa: BLE001 -- diagnostics must never stop a load
            pass


def load(path, *, cuda, context=0, say=lambda text: None):
    """-> a Backend for the .gguf file `path` (the first part of a split model: llama.cpp finds the others beside it).
    A projector file beside it is loaded too, so the model can see pictures."""
    llama_cpp = ensure_llama(say)
    watch_native(llama_cpp, say)
    started = time.time()
    say(f"loading {os.path.basename(str(path))} into memory")
    projector, handler = find_projector(path), None
    if projector:
        try:
            handler = projector_handler(llama_cpp, projector, os.path.basename(str(path)))
        except Exception as exc:                                    # noqa: BLE001 -- the model still writes text without it
            say(f"engine: the picture projector {os.path.basename(projector)} could not be loaded ({exc}); text only")
            projector = ""
    llm = llama_cpp.Llama(model_path=str(path), n_gpu_layers=-1 if cuda else 0, n_ctx=int(context) or 32768, flash_attn=bool(cuda), verbose=False, chat_handler=handler, swa_full=False)
    try:
        gpu = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:                                               # noqa: BLE001
        gpu = False
    if cuda and not gpu:
        say("engine: this llama.cpp build has no graphics-card support, so the model runs on the processor (several times slower); reinstall llama-cpp-python from the CUDA wheel index")
    say(f"engine: llama.cpp loaded {os.path.basename(str(path))}" + (f" with {os.path.basename(projector)}" if projector else "") + f" in {time.time() - started:.0f}s" + (", on the graphics card" if gpu else ", on the processor"))
    backend = Backend(llm, str(path))
    backend.projector, backend.handler = projector, handler
    return backend
