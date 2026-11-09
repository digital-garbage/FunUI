"""Where H3 puts the prompt.

Two capabilities, so prompt markup never learns H3's layout:

* `text_tokenizer(clip)` -- the raw HF tokenizer behind an H3 CLIP, for exact
  character -> token offsets. Tokenizing is free; re-encoding with Qwen3-VL-32B
  to find a phrase would cost a 32B forward pass per phrase.
* `prompt_rows(meta, cond_len, prompt_tokens)` -- (start, end) of the prompt
  inside a conditioning. The prompt is the LAST thing H3's tokenizer appends
  (reference labels and vision blocks come first), and `minimax_token_tags`
  marks text 1, vision 0, so it is the tail of the final run of 1s. Measured
  from the END so labels in front of it (an audio reference's "<Audio n>: ",
  which has no vision block) do not shift it.

The conditioning block leads the packed sequence, so a row inside it is also a
row of the sequence the attention runs over.
"""


def text_tokenizer(clip):
    try:
        tok = getattr(getattr(getattr(clip, "tokenizer", None), "qwen3vl_32b", None),
                      "tokenizer", None)
        if tok is None:
            return None
        tok("probe", add_special_tokens=False, return_offsets_mapping=True)
        return tok
    except Exception:                            # noqa: BLE001 -- not H3's, or no offsets
        return None


def _tags(meta):
    tags = (meta or {}).get("minimax_token_tags")
    if tags is None:
        return None
    try:
        return [int(t) for t in tags.reshape(-1).tolist()]
    except AttributeError:
        return [int(t) for t in tags] if isinstance(tags, (list, tuple)) else None


def prompt_rows(meta, cond_len, prompt_tokens):
    """(start, end) of the prompt in the conditioning, or None if it cannot be
    proven. Without tags the prompt is the final `prompt_tokens` rows."""
    if prompt_tokens <= 0 or cond_len <= 0:
        return None
    tags = _tags(meta)
    if tags is None:
        start = cond_len - prompt_tokens
        return (start, cond_len) if start >= 0 else None
    end = min(len(tags), cond_len)
    while end > 0 and tags[end - 1] != 1:
        end -= 1
    start = end
    while start > 0 and tags[start - 1] == 1:
        start -= 1
    if end - start < prompt_tokens:
        return None                              # a different string was encoded
    return end - prompt_tokens, end
