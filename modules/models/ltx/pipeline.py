"""LTX 2.3 pipelines, as loadable presets.

Built from ComfyUI's own LTX nodes plus FunUI's, the chain the stock templates use:
conditioning carries the frame rate, and the sampler runs the distilled
schedule written out as numbers (`project_sigmas`: the official anchors, never replaced).

Four separate files, as most people keep them: the diffusion model, the video VAE, the audio VAE and the text
encoder(s), each in its own loader (FunUI's own, like the H3 pipeline).

Every `frame_rate` and `fps` is a role at `project.video`, so the project's FPS is the one number
the conditioning, the sound's length and the saved video agree on.
"""

#: The distilled 8-step schedule, from the official ComfyUI LTX 2.x templates.
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def _common(latent_slots, latent_source, positive_source, negative_source, decode_source=("sampler", 0)):
    """What both pipelines share: the four loaders, the prompt, the sampler and the render.
    `latent_slots` hold the latent the sampler starts from and whatever makes it."""
    video = {"at": "project.video", "input": "frame_rate", "label": "FPS", "drives": "fps"}
    return [
        {"id": "model", "group": "Loaders", "node": "FunUIDiffusionModelLoader",
         "inputs": {"weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
        {"id": "vae", "group": "Loaders", "node": "FunUIVAELoader", "inputs": {"dtype": "bfloat16"}},
        {"id": "audio_vae", "group": "Loaders", "node": "FunUIVAELoader", "inputs": {"dtype": "bfloat16"}},
        {"id": "clip", "group": "Loaders", "node": "FunUICLIPLoader", "inputs": {"type": "ltxv"}},
        {"id": "positive", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "generation.prompt", "input": "text", "label": "Prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "project.negative", "input": "text", "label": "Negative prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "conditioning", "group": "Preparation", "node": "LTXVConditioning",
         "roles": [video],
         "inputs": {"positive": ["positive", 0], "negative": ["negative", 0], "frame_rate": 25.0}},
        *latent_slots,
        {"id": "settings", "group": "Preparation", "node": "FunUIModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunUILoadModifiers",
         "inputs": {"model": ["model", 0], "settings": ["settings", 0]}},
        {"id": "sigmas", "group": "Sampling", "node": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        {"id": "sampler", "group": "Sampling", "node": "FunUISampler",
         "roles": [{"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {"model": ["modifiers", 0], "positive": positive_source, "negative": negative_source,
                    "latent": latent_source, "settings": ["settings", 0], "sigmas": ["sigmas", 0],
                    "seed": 0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal",
                    "denoise": 1.0}},
        {"id": "decode", "group": "Render", "node": "FunUIDecode", "inputs": {
            "samples": list(decode_source), "vae": ["vae", 0], "model": ["model", 0], "audio_vae": ["audio_vae", 0]}},
        {"id": "save", "group": "Render", "node": "FunUISaveVideo",
         "roles": [{"at": "project.video", "input": "fps", "label": "FPS", "drives": "fps"}],
         "inputs": {"images": ["decode", 0], "audio": ["decode", 1], "fps": 25.0, "filename_prefix": "FunUI"}},
    ]


def _image_to_video():
    """Image to video, the start taken from the scene's first reference picture (nothing is picked twice). The picture on the
    timeline is the ASPECT SOURCE, as on H3: it gives the clip its shape, the project's Width x Height give the scale. With no
    picture in References, the anchor is bypassed and the clip is made from the prompt alone (text to video)."""
    video = {"at": "project.video", "input": "frame_rate", "label": "FPS", "drives": "fps"}
    latent = [
        {"id": "aspect_source", "group": "Reference media", "node": "FunUILoadMedia",
         # With no picture a plain one of the budget's size stands in, so the shape is the project's own.
         "roles": [{"at": "assets.source_image", "input": "media_id",
                    "blank_like": {"slot": "aspect", "width": "width", "height": "height"}}],
         "inputs": {"media_id": "", "blank_width": 0, "blank_height": 0}},
        {"id": "aspect", "group": "Preparation", "node": "FunUIAspectSize",
         "roles": [{"at": "project.video", "input": "width", "label": "Width"},
                   {"at": "project.video", "input": "height", "label": "Height"},
                   {"at": "project.resolution_basis", "input": "basis"}],
         "inputs": {"image": ["aspect_source", 0], "width": 768, "height": 512, "multiple_of": 32, "basis": "short"}},
        {"id": "latent", "group": "Preparation", "node": "FunUIEmptyLatent",
         "roles": [{"at": "project.video", "input": "length", "label": "Length", "drives": "frames"}, video],
         "inputs": {"model": ["model", 0], "audio_vae": ["audio_vae", 0],
                    "width": ["aspect", 0], "height": ["aspect", 1], "length": 121, "frame_rate": 25.0, "batch_size": 1}},
        {"id": "source_image", "group": "Reference media", "node": "FunUILoadMedia",
         "roles": [{"at": "assets.reference_1", "input": "media_id",
                    # Nothing in References: the anchor is bypassed (the picture it would carry is a plain stand-in).
                    "text_fallback": {"set": [{"slot": "anchor", "input": "bypass", "value": True}],
                                      "size": {"slot": "aspect", "width": "width", "height": "height"}}}],
         "inputs": {"media_id": ""}},
        {"id": "split", "group": "Preparation", "node": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["latent", 0]}},
        {"id": "anchor", "group": "Preparation", "node": "LTXVImgToVideoInplace",
         "inputs": {"vae": ["vae", 0], "image": ["source_image", 1], "latent": ["split", 0],
                    "strength": 1.0, "bypass": False}},
        # Pictures from the clips before (Guide frames, filled per run by the app; nothing in it passes everything through).
        {"id": "guides", "group": "Preparation", "node": "FunUIGuideFrames",
         "inputs": {"positive": ["conditioning", 0], "negative": ["conditioning", 1], "vae": ["vae", 0],
                    "latent": ["anchor", 0], "frames": ""}},
        {"id": "join", "group": "Preparation", "node": "LTXVConcatAVLatent",
         "inputs": {"video_latent": ["guides", 2], "audio_latent": ["split", 1]}},
        # Guides sit at the tail of the sequence while sampling: cut them off before decoding.
        {"id": "split_made", "group": "Render", "node": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sampler", 0]}},
        {"id": "crop", "group": "Render", "node": "LTXVCropGuides",
         "inputs": {"positive": ["guides", 0], "negative": ["guides", 1], "latent": ["split_made", 0]}},
        {"id": "join_made", "group": "Render", "node": "LTXVConcatAVLatent",
         "inputs": {"video_latent": ["crop", 2], "audio_latent": ["split_made", 1]}},
    ]
    return _common(latent, ["join", 0], ["guides", 0], ["guides", 1], decode_source=("join_made", 0))


def _direct_video_to_video():
    """Plain video-to-video: the picked video's frames are encoded and sampled again from part-way down the schedule
    (the tail of the distilled sigmas), so the result keeps the video's layout and motion and changes what the prompt and
    the noise ask for. The later the schedule starts, the more of the video survives.

    The size and length are the video's own (Video input sets them); the sound starts empty and is made with the picture.
    With no video, a plain grey clip of the project's size and length stands in and the whole schedule runs (text to video).
    """
    fps = {"at": "project.video", "input": "frame_rate", "label": "FPS", "drives": "fps"}
    latent = [
        # video_frames: the whole video is fed, not its first frame (see inputs.js).
        {"id": "source_image", "group": "Reference media", "node": "FunUILoadMedia",
         "roles": [{"at": "assets.source_image", "input": "media_id", "video_frames": True, "video_only": True,
                    "text_fallback": {"set": [{"slot": "sigmas", "input": "sigmas", "value": DISTILLED_SIGMAS}],
                                      "round_size": 32, "round_frames": 8}},
                   {"at": "project.video", "input": "blank_width", "label": "Width (no video)"},
                   {"at": "project.video", "input": "blank_height", "label": "Height (no video)"},
                   {"at": "project.video", "input": "blank_frames", "label": "Length (no video)", "drives": "frames"}],
         "inputs": {"media_id": "", "blank_width": 768, "blank_height": 512, "blank_frames": 97}},
        {"id": "size", "group": "Preparation", "node": "GetImageSize", "inputs": {"image": ["source_image", 1]}},
        {"id": "encode", "group": "Preparation", "node": "VAEEncode", "inputs": {"pixels": ["source_image", 1], "vae": ["vae", 0]}},
        {"id": "audio_latent", "group": "Preparation", "node": "LTXVEmptyLatentAudio",
         "roles": [fps], "inputs": {"frames_number": ["size", 2], "frame_rate": 25, "batch_size": 1, "audio_vae": ["audio_vae", 0]}},
        {"id": "latent", "group": "Preparation", "node": "LTXVConcatAVLatent",
         "inputs": {"video_latent": ["encode", 0], "audio_latent": ["audio_latent", 0]}},
    ]
    slots = _common(latent, ["latent", 0], ["conditioning", 0], ["conditioning", 1])
    for x in slots:
        if x["id"] == "sigmas":
            x["inputs"]["sigmas"] = "0.909375, 0.725, 0.421875, 0.0"      # the last three steps of the distilled schedule: about 90% re-noised
    return slots


#: See minimax_h3/pipeline.py: what to pass through when the node's module did not load.
OPTIONAL_NODES = {"FunUIIdentityTransfer": {"pass": {"0": "model", "1": "positive", "2": "negative"}},
                  "FunUIGuideFrames": {"pass": {"0": "positive", "1": "negative", "2": "latent"}}}


def _add_aspect(slots):
    """Image to Video from References saved before the timeline picture became its aspect source: the two slots that give the
    shape are put in front of the latent, and the latent reads the size from them (the size typed there stays the budget)."""
    ids = {x.get("id"): x for x in slots}
    latent, start = ids.get("latent"), ids.get("source_image")
    if "aspect" in ids or not latent or latent.get("node") != "FunUIEmptyLatent" or not start or not any(
            r.get("at") == "assets.reference_1" and r.get("text_fallback") for r in start.get("roles") or []):
        return slots
    fresh = {x["id"]: x for x in _image_to_video()}
    aspect = {**fresh["aspect"], "inputs": {**fresh["aspect"]["inputs"], **{k: latent["inputs"][k] for k in ("width", "height") if isinstance(latent["inputs"].get(k), int)}}}
    out = []
    for x in slots:
        if x is latent:
            out += [fresh["aspect_source"], aspect]
            x = {**x, "inputs": {**x["inputs"], "width": ["aspect", 0], "height": ["aspect", 1]},
                 "roles": [r for r in x.get("roles") or [] if r.get("input") not in ("width", "height")]}
        elif x is start:
            x = {**x, "roles": [dict(r, text_fallback={**r["text_fallback"], "size": fresh["source_image"]["roles"][0]["text_fallback"]["size"]}) if r.get("text_fallback") else r
                                for r in x.get("roles") or []]}
        out.append(x)
    return out


def _add_text_fallback(slots):
    """An Image to Video saved before text to video was the fallback: its start picture is required, so a run with nothing fed
    in could not start. The start slot now says what to change when there is none (bypass the anchor, a plain picture of the
    latent's size stands in), as the shipped preset does."""
    ids = {x.get("id"): x for x in slots}
    start, anchor, latent = ids.get("source_image"), ids.get("anchor"), ids.get("latent")
    if not start or not anchor or anchor.get("node") != "LTXVImgToVideoInplace" or not latent or start.get("node") != "FunUILoadMedia":
        return slots
    roles = start.get("roles") or []
    if any(r.get("text_fallback") for r in roles) or not any(r.get("input") == "media_id" for r in roles):
        return slots
    if "bypass" not in (anchor.get("inputs") or {}):
        return slots
    size = {"slot": "aspect", "width": "width", "height": "height"} if "aspect" in ids else {"slot": "latent", "width": "width", "height": "height"}
    fallback = {"set": [{"slot": "anchor", "input": "bypass", "value": True}], "size": size}
    new = [dict(r, text_fallback=fallback) if r.get("input") == "media_id" else r for r in roles]
    return [dict(x, roles=new) if x is start else x for x in slots]


def upgrade(slots, _outputs=None):
    return _add_text_fallback(_add_aspect(_upgrade_video(slots)))


def _upgrade_video(slots):
    """Pipelines saved by an earlier build, brought up to date:
    * the picture inputs read the loader's first output (always one frame): a start slot that takes whole videos feeds them
      from the frames output now;
    * the direct video-to-video pipeline, saved before its start slot said it takes videos only."""
    start = next((s for s in slots if s.get("id") == "source_image" and s.get("node") == "FunUILoadMedia"
                  and any(r.get("video_frames") for r in s.get("roles") or [])), None)
    if not start:
        return slots
    encoded = any(s.get("node") == "VAEEncode" and (s.get("inputs") or {}).get("pixels") in (["source_image", 0], ["source_image", 1]) for s in slots)
    out = []
    for s in slots:
        inputs = {k: (["source_image", 1] if isinstance(v, (list, tuple)) and list(v) == ["source_image", 0] else v)
                  for k, v in (s.get("inputs") or {}).items()}
        item = dict(s, inputs=inputs)
        if s is start and encoded:
            item["roles"] = [dict(r, video_only=True) if r.get("video_frames") else r for r in s.get("roles") or []]
        out.append(item)
    return out


def optional_nodes():
    return OPTIONAL_NODES


def presets():
    return [{"id": "ltx23_image_to_video_references", "title": "LTX 2.3 · Image to Video, from References",
             "slots": _image_to_video()},
            {"id": "ltx23_video_to_video_direct", "title": "LTX 2.3 · Video to Video, direct",
             "slots": _direct_video_to_video()}]


PROVIDES = {"pipeline_presets": presets}
