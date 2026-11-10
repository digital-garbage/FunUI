"""The two markup nodes. See the module docstring for the shape."""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ..._core import control as control_mod, dit_hooks, log, patching, registry as registry_mod, schema as schema_mod
from . import markup as mk

MARKUP = io.Custom("FUNUI_MARKUP")
KEY = "funui_markup.prompt"
PHRASES = "funui_markup_phrases"   # the transformer_options key the phrase probe reads
SIMUL = "funui_simultaneous"        # ... and the one Step merging reads: one whole conditioning per option, not yet refined by the model


def _ask(capability, *args):
    """The first model module with an answer, or None."""
    for _spec, provider in registry_mod.current().providers(capability):
        got = provider(*args)
        if got is not None:
            return got
    return None


def _say(message):
    log.once(f"prompt_markup:{message}", log.ALERT, "FunUI Prompt Markup", message)


def _alternatives(settings):
    """-> (how the main prompt reads an [A|B], whether A and B are also read separately), from Alternating prompts."""
    spec = registry_mod.current().specs.get("prompt_alternatives")
    if spec is None or control_mod.all_off(settings):
        return "while", False
    clean, _problems = schema_mod.check_values(spec, (settings or {}).get("prompt_alternatives"))
    mode = clean.get("main")
    return (mode if mode in mk.MAIN_MODES else "while"), bool(clean.get("separate"))


class FunUIPromptMarkup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUIPromptMarkup",
            display_name="FunUI Prompt Markup",
            category="FunUI/Conditioning",
            description="Take (word:1.5) and [a|b] out of a prompt, so the encoder sees clean text "
                        "and the markup can be applied properly.",
            inputs=[io.String.Input("text", multiline=True, default=""),
                    io.Custom("FUNUI_SETTINGS").Input("settings", optional=True)],
            outputs=[io.String.Output(display_name="clean"),
                     MARKUP.Output(display_name="markup"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, text: str, settings=None) -> io.NodeOutput:
        main, separate = _alternatives(settings)
        parsed = mk.parse(text or "", main=main)
        parsed["separate"] = separate
        n = {"weighted": len(parsed["weighted"]), "alternative": len(parsed["variants"])}
        status = (", ".join(f"{v} {k}" for k, v in n.items() if v) or "no markup")
        return io.NodeOutput(parsed["clean"], parsed, status)


def _variant_rows(rows, clip, tokenizer, texts):
    """The prompt rows of each option's own prompt, brought to the length of `rows` (the prompt as encoded) --
    [1, n, D] each, or None when one of them cannot be placed."""
    start, end = rows
    out = []
    for text in texts:
        encoded = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        cond, meta = encoded[0][0], encoded[0][1]
        n = len(tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"])
        placed = _ask("prompt_rows", meta, int(cond.shape[1]), n)
        if placed is None:
            return None
        got = cond[:, placed[0]:placed[1], :]
        if got.shape[1] != end - start:
            # Linear along the token axis: a smoothing, not a word alignment.
            got = F.interpolate(got.float().transpose(1, 2), size=end - start, mode="linear",
                                align_corners=True).transpose(1, 2).to(got.dtype)
        out.append(got)
    return out


class FunUIApplyPromptMarkup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUIApplyPromptMarkup",
            display_name="FunUI Apply Prompt Markup",
            category="FunUI/Conditioning",
            description="Apply the markup FunUI Prompt Markup took out: weights on the model, and one "
                        "prompt per [a|b] option for Step merging to use.",
            inputs=[io.Model.Input("model"), io.Clip.Input("clip"),
                    io.Conditioning.Input("positive"), io.Latent.Input("latent"),
                    MARKUP.Input("markup")],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, model, clip, positive, latent, markup) -> io.NodeOutput:
        patched = patching.clone(model)
        patching.strip(patched, "funui_markup.")
        dit_hooks.publish(patched, SIMUL, None)
        weighted = (markup or {}).get("weighted")
        variants = (markup or {}).get("variants") if (markup or {}).get("separate") else None
        if not (weighted or variants):
            return io.NodeOutput(patched, positive, "no markup")

        def refuse(why):
            _say(f"markup NOT applied: {why}. The prompt still went in, without it.")
            return io.NodeOutput(patched, positive, f"not applied: {why}")

        tokenizer = _ask("text_tokenizer", clip)
        if tokenizer is None:
            return refuse("this text encoder is not one a model module can place words in")
        clean = markup["clean"]
        offsets = list(tokenizer(clean, add_special_tokens=False,
                                 return_offsets_mapping=True)["offset_mapping"])
        cond, meta = positive[0][0], positive[0][1]
        cond_len = int(cond.shape[1])
        rows = _ask("prompt_rows", meta, cond_len, len(offsets))
        if rows is None:
            return refuse("the encoded prompt does not match this text -- was the clean "
                          "text from Prompt Markup what got encoded?")
        said = []
        if markup.get("extra"):
            _say(f"[a|b] takes {mk.MAX_OPTIONS} options; the {markup['extra']} after them are ignored")
        if variants:
            got = _variant_rows(rows, clip, tokenizer, variants)
            if got is None:
                _say("a [a|b] option's prompt could not be placed; the options are off")
            else:
                whole = []
                for rows_k in got:                     # the full conditioning, as encoded, with this option's prompt rows
                    raw = cond.clone()
                    raw[:, rows[0]:rows[1]] = rows_k.to(device=raw.device, dtype=raw.dtype)
                    whole.append(raw)
                dit_hooks.publish(patched, SIMUL, {"options": whole, "cond_len": cond_len})
                said.append(f"{len(got)} simultaneous option(s)")
        w_spans = mk.token_spans(offsets, weighted or [])
        if w_spans:
            cls._install(patched, w_spans, rows[0], cond_len)
            # Where the phrases sit, for anything that wants to look at them (the phrase probe).
            dit_hooks.publish(patched, PHRASES, {
                "base": rows[0], "cond_len": cond_len, "spans": [(a, b) for a, b, *_ in w_spans]})
            said.append(f"{len(w_spans)} weight(s)")
        return io.NodeOutput(patched, positive, ", ".join(said) or "nothing placeable")

    @staticmethod
    def _install(patcher, w_spans, base, cond_len):
        cache = {}

        def override(func, q, k, v, heads, *args, mask=None, **kwargs):
            def run(m):
                return func(q, k, v, heads, *args, mask=m, **kwargs)

            # Only the packed self-attention: the prompt plus the streams after it.
            # A sequence exactly as long as the prompt is the text-only token
            # refiner, which must never see the bias.
            if k.ndim != 4 or q.shape[2] != k.shape[2] or k.shape[2] <= cond_len:
                return run(mask)
            try:
                key = (int(k.shape[2]), k.device, k.dtype)
                if key not in cache:
                    cache[key] = mk.key_bias(w_spans, base, *key)
                bias = cache[key]
                return run(mask if bias is None else (bias if mask is None else mask + bias))
            except Exception as exc:             # noqa: BLE001 -- markup never costs the step
                _say(f"markup failed inside attention and was dropped ({exc})")
                return run(mask)

        dit_hooks.add_attention_override(patcher, KEY, override)
