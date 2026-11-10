"""Prompt markup -- (word:1.5) and [a|b]. A node-only module.

Two nodes, because the markup has to be split around the encoder: it must come
OUT before the encoder sees the text (Qwen reads "(word:1.5)" as punctuation in
your sentence), and it can only be applied AFTER, once there is a conditioning
to find the words in and a model to bias.

    text -> Prompt Markup -> clean -> (any H3 encode node) -> positive
                          -> markup ---------------------> Apply Prompt Markup
                                                           (+ model, clip, latent)

Where the words sit is the MODEL's business: model modules announce `text_tokenizer`
and `prompt_rows`. With
none that recognise the encoder, the markup is stripped and said to be unused --
never silently ignored.

The weight is an attention BIAS, and SLA runs any biased call dense, so a run
with weights runs dense attention.

`[a|b]` (two things that happen at once) changes nothing by itself: it leaves one prompt per option
in the model for Step merging to answer and merge.
"""

from .nodes import FunUIApplyPromptMarkup, FunUIPromptMarkup

ID = "conditioning_prompt_markup"
TITLE = "Prompt markup"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "proven"

NODES = [FunUIPromptMarkup, FunUIApplyPromptMarkup]
