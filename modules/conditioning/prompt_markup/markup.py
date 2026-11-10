"""Prompt markup: the text part. Pure -- strings in, spans out, tensors built.

Two markups, stripped in ONE left-to-right pass so every span indexes the
same clean text (stripping them one after another left the first list's
offsets pointing into a string that no longer existed):

* `(phrase:1.5)` / `[phrase:1.5]` -- weight. A logit bias of log(w) on that
  phrase's key positions: softmax(logits + log w) multiplies that key's share
  of the attention by w, which is what a weight is supposed to mean.
* `[A|B]` -- simultaneous actions: two things that happen at the same time.
  The prompt the model normally reads holds both ("A while B"); `variants` holds
  one prompt per option with the others left out, for the sampler to answer
  separately and merge (Step merging > Simultaneous actions).

Weighting only exists because the text is IN the attention sequence (H3's
single packed stream). Scaling embeddings would not work: Qwen's rows are
contextual hidden states, with no per-word magnitude to turn up.
"""

import math
import re

_NUM = r"-?(?:\d+\.\d*|\.\d+|\d+)"
_WEIGHTED = re.compile(r"(?<!\\)\(([^():]*?):\s*(" + _NUM + r")\s*\)")
_BRACKET_WEIGHTED = re.compile(r"(?<!\\)\[([^\[\]@|]*?):\s*(" + _NUM + r")\s*\]")
_ALTERNATIVES = re.compile(r"(?<!\\)\[([^\[\]|]+(?:\|[^\[\]|]+)+)\]")
_MARKUP = re.compile("|".join(p.pattern for p in (_ALTERNATIVES, _WEIGHTED, _BRACKET_WEIGHTED)))

# The prompt carries each option in turn, joined like this (what the model reads outside the merged steps).
JOINER = " while "
MAX_OPTIONS = 2
# Below this a weight means "remove": log(0) is -inf, which is NaN the moment a
# query attends to nothing else.
MIN_WEIGHT = 1e-3
MASKED_BIAS = -30.0
# Past this a phrase wins every query regardless of content: the prompt reads as
# collapsing onto one word.
MAX_ABS_BIAS = 6.0
# Overlapping spans (a phrase and one of its own words) add, bounded at this
# multiple of the strongest single span -- a lone (word:2.0) still means 2.0.
OVERLAP_HEADROOM = 1.5


def _options(raw):
    return [o.strip() for o in raw.split("|") if o.strip()]


def parse(text, pick=None):
    """-> {"clean", "weighted": [(c0, c1, w)], "variants": [clean text per option], "extra": n}, char spans
    into `clean`. `clean` holds every option joined; with `pick` k it holds option k of each [A|B] instead
    (how the variants are made)."""
    out = {"clean": text or "", "weighted": [], "variants": [], "extra": 0}
    if not text or ("(" not in text and "[" not in text):
        return out
    pieces, pos, width, found = [], 0, 0, 0
    for m in _MARKUP.finditer(text):
        pieces.append(text[pos:m.start()])
        width += len(pieces[-1])
        if m.group(1) is not None:
            opts = _options(m.group(1))
            if len(opts) < 2:                      # "[x|]": nothing to be simultaneous with
                phrase = opts[0] if opts else ""
            else:
                found = max(found, len(opts))
                phrase = (JOINER.join(opts) if pick is None else opts[min(pick, len(opts) - 1)])
        else:
            phrase = m.group(2) if m.group(2) is not None else m.group(4)
            weight = m.group(3) if m.group(3) is not None else m.group(5)
            out["weighted"].append((width, width + len(phrase), float(weight)))
        pieces.append(phrase)
        width += len(phrase)
        pos = m.end()
    pieces.append(text[pos:])
    out["clean"] = "".join(pieces)
    if found >= 2 and pick is None:
        out["variants"] = [parse(text, pick=k)["clean"] for k in range(min(found, MAX_OPTIONS))]
        out["extra"] = max(0, found - MAX_OPTIONS)
    return out


def token_spans(offsets, char_spans):
    """Char spans -> (tok0, tok1, *rest) using a tokenizer's offset mapping."""
    out = []
    for c0, c1, *rest in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if a < c1 and b > c0]
        if toks:
            out.append((toks[0], toks[-1] + 1, *rest))
    return out


def bias_value(weight):
    if weight <= MIN_WEIGHT:
        return MASKED_BIAS
    return max(-MAX_ABS_BIAS, min(MAX_ABS_BIAS, math.log(weight)))


def key_bias(spans, base, seq_len, device, dtype):
    """[1, 1, 1, seq_len] additive bias for (tok0, tok1, w) spans, prompt at `base`."""
    import torch
    bias, strongest = None, 0.0
    for t0, t1, weight in spans:
        b = bias_value(weight)
        if b == 0.0 or t1 <= t0:
            continue
        if bias is None:
            bias = torch.zeros(1, 1, 1, seq_len, device=device, dtype=dtype)
        bias[..., base + t0:base + t1] += b
        strongest = max(strongest, abs(b))
    if bias is not None:
        limit = min(MAX_ABS_BIAS, strongest * OVERLAP_HEADROOM)
        bias.clamp_(min=-limit, max=limit)
    return bias
