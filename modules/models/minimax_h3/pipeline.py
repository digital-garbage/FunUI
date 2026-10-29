"""The MiniMax H3 · Reference to Video pipeline, as a loadable preset.

Not the app's one true default -- `modules/system/pipeline` already owns that,
and a second module silently claiming `default_pipeline` would make which one
actually starts depend on load order. This is offered as a PRESET instead: a
slot list a person picks from the pipeline window, same shape as the default,
loaded into the very same editable graph (`core/graph.py`'s wire/replace/remove
apply to it exactly as they do to anything else).

The chain mirrors what this app's own H3 user actually runs by hand today:
loaders, MiniMaxH3SigmaShift patching the model's flow shifts, an
ImageTransformKJ node turning a dropped-in picture into canvas-multiple
width/height ints -- but the picture supplies only its ASPECT RATIO, not the
resolution. The actual pixel budget is target_width*target_height, role-
mounted at "project.video" exactly like the default pipeline's own
width/height: a project setting, typed once, not implied by whatever a phone
camera happened to produce. "total_pixels" keep_proportion is what makes that
true -- it fits the image's own aspect ratio into that budget's area, rather
than using the image's raw size. Those two resulting ints feed
MiniMaxH3ReferenceToVideo's width/height, and Reference-to-Video's own
conditioning+latent feeds a plain sampler at CFG 1 (H3 wants no negative
guidance) into FunUIDecode and a video/audio mux.

The image and each reference are FunUILoadMedia nodes: fed by a `media_id`
this app sets from what is picked in the Media bin, at "assets.*" roles that
mount no on-screen field of their own (nothing here `offer()`s an `assets.*`
region) -- the same "a role naming a place nobody offers is simply not shown"
rule as everywhere else, used here on purpose so boot.js can still find these
slots BY ROLE (never by this preset's own slot ids) without the app rendering
a raw text box asking someone to type a media id by hand.
"""

#: How many references (of any kind) this preset has a slot for: R2V takes up to 9 pictures, 3 videos
#: (each with its soundtrack) and 3 sounds. More slots only add optional ones, and a saved project gets
#: the new ones when it opens.
MAX_REFERENCES = 15


def h3_reference_to_video():
    slots = [
        {"id": "model", "group": "Loaders", "node": "FunUIDiffusionModelLoader", "inputs": {
            "weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
        {"id": "clip", "group": "Loaders", "node": "FunUICLIPLoader",
         "inputs": {"type": "minimax"}},
        {"id": "vae", "group": "Loaders", "node": "FunUIVAELoader",
         "inputs": {"dtype": "bfloat16"}},
        {"id": "audio_vae", "group": "Loaders", "node": "FunUIVAELoader",
         "inputs": {"dtype": "bfloat16"}},

        # Always present, does nothing until a LoRA is picked; chain more through Models & Pipeline.
        {"id": "lora", "group": "Loaders", "node": "FunUILoraLoader",
         "inputs": {"model": ["model", 0], "lora_name": "None", "strength_model": 1.0}},
        {"id": "shift", "group": "Sampling", "node": "MiniMaxH3SigmaShift",
         "roles": [{"at": "generation.model", "input": "shift_video", "label": "Video shift"},
                   {"at": "generation.model", "input": "shift_audio", "label": "Audio shift"}],
         "inputs": {"model": ["lora", 0], "shift_video": 12.0, "shift_audio": 3.0}},

        {"id": "source_image", "group": "Reference media", "node": "FunUILoadMedia",
         # Unhosted: see the module docstring. boot.js finds this slot by this
         # role's name and sets `media_id` from the scene's own picked image,
         # the same way "generation.prompt" is found by name and set from the
         # scene's own text -- neither is addressed by this preset's slot id.
         # `blank_like`: with no picture (text-only), buildInputs gives this node the size of that
         # slot's target_width/height, so a plain grey stand-in carries the project's own shape.
         "roles": [{"at": "assets.source_image", "input": "media_id",
                    "blank_like": {"slot": "image_transform", "width": "target_width", "height": "target_height",
                                     "unsend": {"slot": "enhance", "input": "use_image"}}}],
         "inputs": {"media_id": "", "blank_width": 0, "blank_height": 0}},
        {"id": "image_transform", "group": "Reference media", "node": "ImageTransformKJ",
         # target_width/height are the PIXEL BUDGET, set in Project settings
         # like any other project resolution -- not the dropped picture's own
         # size. "total_pixels" is what makes that budget win: it takes
         # target_width*target_height as an area and fits the image's own
         # aspect ratio into it, so the picture supplies the SHAPE and the
         # project supplies the SCALE. A 12-megapixel phone photo dropped in
         # must not become the generation's actual resolution.
         "roles": [{"at": "project.video", "input": "target_width", "label": "Width"},
                   {"at": "project.video", "input": "target_height", "label": "Height"}],
         "inputs": {
            "image": ["source_image", 0], "target_width": 1344, "target_height": 768,
            "upscale_method": "lanczos",
            "keep_proportion": "total_pixels",
            "divisible_by": 32,
            "extra_padding": "disabled",
            "invert_crop": "disabled",
            "bboxes": ""}},

        # Off by default: the prompt passes straight through and the model never
        # runs. Before the markup node because a rewrite changes the words the
        # markup points at -- (word:1.5) typed with this on is sent to the model
        # as text, and the node says so.
        {"id": "enhance", "group": "Preparation", "node": "FunUIEnhancePrompt",
         "roles": [{"at": "generation.prompt", "input": "text", "label": "Prompt"},
                   {"at": "generation.prompt", "input": "enabled", "label": "Enhance prompt"}],
         "inputs": {"clip": ["clip", 0], "text": "", "enabled": False,
                    "image": ["source_image", 0]}},
        # Off by default (its own switches in Settings): writes cut times, views and camera moves into [Shot N] blocks.
        {"id": "shot_camera", "group": "Preparation", "node": "FunUIShotCamera",
         "roles": [{"at": "project.video", "input": "length", "label": "Length", "drives": "frames"}],
         "inputs": {"text": ["enhance", 0], "settings": ["settings", 0], "length": 124, "frame_rate": 24.0}},
        {"id": "markup", "group": "Preparation", "node": "FunUIPromptMarkup",
         "inputs": {"text": ["shot_camera", 0]}},
        {"id": "r2v", "group": "Preparation", "node": "MiniMaxH3ReferenceToVideo",
         "roles": [{"at": "project.video", "input": "length", "label": "Length", "drives": "frames"}],
         "inputs": {
             "clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["audio_vae", 0],
             "prompt": ["markup", 0], "width": ["image_transform", 4], "height": ["image_transform", 5],
             "length": 124, "ref_image_size": "match"}},
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         # H3 samples at CFG 1, where a negative prompt does nothing -- wired
         # only because FunUISampler requires SOME conditioning here. The project's Negative
         # prompt lands here so Shadow negative and Negative erase have words to work from.
         "roles": [{"at": "project.negative", "input": "text", "label": "Negative prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},

        {"id": "settings", "group": "Preparation", "node": "FunUIModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunUILoadModifiers", "inputs": {
            "model": ["shift", 0], "settings": ["settings", 0]}},

        {"id": "markup_apply", "group": "Preparation", "node": "FunUIApplyPromptMarkup",
         "inputs": {"model": ["modifiers", 0], "clip": ["clip", 0], "positive": ["r2v", 0],
                    "latent": ["r2v", 1], "markup": ["markup", 1]}},

        # Strength 0 = untouched. Above it, the negative prompt (dead at CFG 1) is taken out of the words.
        {"id": "erase", "group": "Preparation", "node": "FunUINegativeErase",
         "inputs": {"positive": ["markup_apply", 1], "negative": ["negative", 0], "strength": 0.0,
                    "mode": "project", "keep_size": True}},

        {"id": "sampler", "group": "Sampling", "node": "FunUISampler",
         "roles": [{"at": "generation.sampling", "input": "steps", "label": "Steps"},
                   {"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.sampling", "input": "scheduler", "label": "Scheduler"},
                   {"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {
             "model": ["markup_apply", 0], "positive": ["erase", 0],
             "negative": ["negative", 0], "latent": ["r2v", 1], "settings": ["settings", 0],
             "seed": 0, "steps": 20, "cfg": 1.0,
             "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},

        {"id": "decode", "group": "Render", "node": "FunUIDecode", "inputs": {
            "samples": ["sampler", 0], "vae": ["vae", 0], "model": ["shift", 0],
            "audio_vae": ["audio_vae", 0]}},
        {"id": "cut", "group": "Render", "node": "FunUICutOpening",
         "roles": [{"at": "project.video", "input": "frames", "label": "Cut opening frames"}],
         "inputs": {"images": ["decode", 0], "audio": ["decode", 1], "frames": 0, "fps": 24.0}},
        {"id": "save", "group": "Render", "node": "FunUISaveVideo", "inputs": {
            "images": ["cut", 0], "audio": ["cut", 1], "fps": 24.0, "filename_prefix": "FunUI"}},
    ]

    for n in range(1, MAX_REFERENCES + 1):
        slots.append({
            "id": f"ref_source_{n}", "group": "Reference media", "node": "FunUILoadMedia",
            # `wireTo` is not read by core or by graph.py -- it is this
            # preset's own data, read by boot.js at queue time (see
            # queueInputs()'s handling of "assets.reference_N" there) to know
            # where a reference's IMAGE output should land WITHOUT boot.js
            # having to know "r2v" or "ref_images" itself. Not wired here in
            # the STATIC slot list -- see the module docstring above: an
            # unwired optional autogrow slot is what "this reference is not
            # in use" actually looks like to MiniMaxH3ReferenceToVideo,
            # absent from its `ref_images` dict entirely, not an empty image
            # degrading its output. Only a scene that actually has this many
            # references gets this wire added, per-run.
            "roles": [{"at": f"assets.reference_{n}", "input": "media_id",
                       "wireTo": {"slot": "r2v", "input": f"ref_images.ref_image_{n - 1}"}}],
            "inputs": {"media_id": ""}})

    return slots


def h3_reference_to_video_two_stage():
    """The same pipeline run in two stages of different size (the cross-resolution schedule of NVIDIA's Sol-H3).

    The model's own schedule is cut in two. The first steps (where the shot's layout and motion are
    decided) run on a smaller picture, the model's clean guess at that point is resized to the full
    size, and the last steps run there, so most of the early work costs a fraction. References and
    text ride along unchanged: only the empty latent differs in size. Untested on a GPU.
    """
    slots = h3_reference_to_video()
    slots = [s for s in slots if s["id"] != "sampler"]
    at = next(i for i, s in enumerate(slots) if s["id"] == "decode")
    sampling_roles = [{"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                      {"at": "generation.seed", "input": "seed", "label": "Seed"}]
    slots[at:at] = [
        {"id": "stage_size", "group": "Sampling", "node": "FunUIStageSize",
         "roles": [{"at": "generation.sampling", "input": "scale", "label": "First stage size"}],
         "inputs": {"width": ["image_transform", 4], "height": ["image_transform", 5], "scale": 0.7}},
        {"id": "latent_low", "group": "Sampling", "node": "FunUIEmptyLatent",
         "roles": [{"at": "project.video", "input": "length", "label": "Length", "drives": "frames"}],
         "inputs": {"model": ["model", 0], "width": ["stage_size", 0], "height": ["stage_size", 1],
                    "length": 124, "batch_size": 1, "audio_vae": ["audio_vae", 0], "frame_rate": 24.0}},
        {"id": "schedule", "group": "Sampling", "node": "BasicScheduler",
         "roles": [{"at": "generation.sampling", "input": "steps", "label": "Steps"},
                   {"at": "generation.sampling", "input": "scheduler", "label": "Scheduler"}],
         "inputs": {"model": ["markup_apply", 0], "scheduler": "normal", "steps": 4, "denoise": 1.0}},
        {"id": "split", "group": "Sampling", "node": "SplitSigmas",
         "roles": [{"at": "generation.sampling", "input": "step", "label": "Steps in the first stage"}],
         "inputs": {"sigmas": ["schedule", 0], "step": 2}},
        {"id": "sampler_low", "group": "Sampling", "node": "FunUISampler",
         "roles": sampling_roles,
         "inputs": {
             "model": ["markup_apply", 0], "positive": ["erase", 0], "negative": ["negative", 0],
             "latent": ["latent_low", 0], "settings": ["settings", 0], "sigmas": ["split", 0],
             "seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        {"id": "resample", "group": "Sampling", "node": "FunUILatentResample",
         "inputs": {"latent": ["sampler_low", 2], "operation": "upscale", "scale": 1.5,
                    "positive": ["erase", 0], "negative": ["negative", 0],
                    "target_width": ["image_transform", 4], "target_height": ["image_transform", 5]}},
        {"id": "sampler", "group": "Sampling", "node": "FunUISampler",
         "roles": sampling_roles,
         "inputs": {
             "model": ["markup_apply", 0], "positive": ["resample", 1], "negative": ["resample", 2],
             "latent": ["resample", 0], "settings": ["settings", 0], "sigmas": ["split", 1],
             "seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
    ]
    return slots


def h3_reference_to_video_second_pass():
    """The same pipeline, then a second pass: resample between passes (sharpen
    by default) and a second sampler running a typed schedule as written.

    The typed schedule's FIRST value is the strength (0.4 polishes, 0.8 reworks
    the shot). The one here is a neutral starting point, not a validated one --
    v4's good second passes came from a schedule the user typed in.
    """
    slots = h3_reference_to_video()
    for slot in slots:
        if slot["id"] == "decode":
            slot["inputs"]["samples"] = ["pass2", 0]
    at = next(i for i, s in enumerate(slots) if s["id"] == "decode")
    slots[at:at] = [
        {"id": "upscaler", "group": "Second pass", "node": "FunUILatentUpscalerLoader",
         "inputs": {}},
        {"id": "resample", "group": "Second pass", "node": "FunUILatentResample",
         "inputs": {"latent": ["sampler", 0], "upscaler": ["upscaler", 0],
                    "operation": "sharpen", "scale": 2.0,
                    "positive": ["erase", 0], "negative": ["negative", 0]}},
        {"id": "pass2_sigmas", "group": "Second pass", "node": "ManualSigmas",
         "inputs": {"sigmas": "0.4, 0.3, 0.2, 0.1, 0.0"}},
        {"id": "pass2", "group": "Second pass", "node": "FunUISampler",
         "roles": [{"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {
             "model": ["markup_apply", 0], "positive": ["resample", 1],
             "negative": ["resample", 2], "latent": ["resample", 0],
             "settings": ["settings", 0], "sigmas": ["pass2_sigmas", 0],
             "seed": 0, "steps": 4, "cfg": 1.0,
             "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
    ]
    return slots


#: Nodes this pipeline can run without, and what to pass through when their module did not load:
#: {node class: {"pass": {output index: input name}}}. Asked for by core (graph.bypass_missing), so saved pipelines get it too.
OPTIONAL_NODES = {
    "FunUIEnhancePrompt": {"pass": {"0": "text"}},
    "FunUIShotCamera": {"pass": {"0": "text"}},
    "FunUIPromptMarkup": {"pass": {"0": "text"}},
    "FunUIApplyPromptMarkup": {"pass": {"0": "model", "1": "positive"}},
    "FunUINegativeErase": {"pass": {"0": "positive"}},
    "FunUICutOpening": {"pass": {"0": "images", "1": "audio"}},
}


def optional_nodes():
    return OPTIONAL_NODES


def presets():
    return [{"id": "minimax_h3_reference_to_video",
             "title": "MiniMax H3 · Reference to Video",
             "slots": h3_reference_to_video()},
            {"id": "minimax_h3_reference_to_video_two_stage",
             "title": "MiniMax H3 · Reference to Video, two stages (test)",
             "slots": h3_reference_to_video_two_stage()}]
    # Held back, not offered: h3_reference_to_video_second_pass() builds clean but was never verified on a GPU
    # (it needs a compatible H3 latent upscaler file). Add it back here once it has been.


PROVIDES = {"pipeline_presets": presets}
