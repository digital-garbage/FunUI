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


def _base(image_to_video: bool, as_guide: bool = False, ic_lora: bool = False):
    video = {"at": "project.video", "input": "frame_rate", "label": "FPS", "drives": "fps"}
    latent_source = ["latent", 0]
    model_source, positive_source, negative_source = ["modifiers", 0], ["conditioning", 0], ["conditioning", 1]
    slots = [
        {"id": "model", "group": "Loaders", "node": "FunUIDiffusionModelLoader",
         "inputs": {"weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
        {"id": "vae", "group": "Loaders", "node": "FunUIVAELoader", "inputs": {"dtype": "bfloat16"}},
        {"id": "audio_vae", "group": "Loaders", "node": "FunUIVAELoader", "inputs": {"dtype": "bfloat16"}},
        {"id": "clip", "group": "Loaders", "node": "FunUICLIPLoader", "inputs": {"type": "ltxv"}},

        *_ic_lora(ic_lora),
        {"id": "positive", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "generation.prompt", "input": "text", "label": "Prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "project.negative", "input": "text", "label": "Negative prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "conditioning", "group": "Preparation", "node": "LTXVConditioning",
         "roles": [video],
         "inputs": {"positive": ["positive", 0], "negative": ["negative", 0], "frame_rate": 25.0}},

        {"id": "latent", "group": "Preparation", "node": "FunUIEmptyLatent",
         "roles": [{"at": "project.video", "input": "width", "label": "Width"},
                   {"at": "project.video", "input": "height", "label": "Height"},
                   {"at": "project.video", "input": "length", "label": "Length", "drives": "frames"},
                   video],
         "inputs": {"model": ["model", 0], "audio_vae": ["audio_vae", 0],
                    "width": 768, "height": 512, "length": 121, "frame_rate": 25.0, "batch_size": 1}},
    ]
    if image_to_video:
        slots += [
            {"id": "source_image", "group": "Reference media", "node": "FunUILoadMedia",
             # video_frames: a video picked as the start is fed whole (prepared as Video input says), not as its first frame.
             "roles": [{"at": "assets.source_image", "input": "media_id", "video_frames": True}],
             "inputs": {"media_id": ""}},
            {"id": "split", "group": "Preparation", "node": "LTXVSeparateAVLatent",
             "inputs": {"av_latent": ["latent", 0]}},
        ]
        if as_guide:
            # The picture steers through guide attention at frame 0 and is NOT the clip's first frame:
            # the latent stays empty, so the opening is free to move. Strength is the slot's own number.
            slots += [
                {"id": "anchor", "group": "Preparation", "node": "LTXVAddGuide",
                 "inputs": {"positive": ["conditioning", 0], "negative": ["conditioning", 1], "vae": ["vae", 0],
                            "latent": ["split", 0], "image": ["source_image", 1], "frame_idx": 0, "strength": 1.0 if ic_lora else 0.35,
                            **({"iclora_parameters": ["ic_params", 0]} if ic_lora else {})}},
                {"id": "join", "group": "Preparation", "node": "LTXVConcatAVLatent",
                 "inputs": {"video_latent": ["anchor", 2], "audio_latent": ["split", 1]}},
            ]
            guided = (["anchor", 0], ["anchor", 1])
        else:
            slots += [
                {"id": "anchor", "group": "Preparation", "node": "LTXVImgToVideoInplace",
                 "inputs": {"vae": ["vae", 0], "image": ["source_image", 1], "latent": ["split", 0],
                            "strength": 1.0, "bypass": False}},
                {"id": "join", "group": "Preparation", "node": "LTXVConcatAVLatent",
                 "inputs": {"video_latent": ["anchor", 0], "audio_latent": ["split", 1]}},
            ]
            guided = (["conditioning", 0], ["conditioning", 1])
        latent_source = ["join", 0]
        positive_source, negative_source = guided
    slots += [
        {"id": "settings", "group": "Preparation", "node": "FunUIModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunUILoadModifiers",
         "inputs": {"model": ["ic_lora", 0] if ic_lora else ["model", 0], "settings": ["settings", 0]}},
    ]
    if image_to_video:
        slots += [
            # A reference face carried into the clip. The picture is the scene's first reference
            # (assets.reference_1): with none picked this node passes everything through untouched.
            {"id": "identity", "group": "Preparation", "node": "FunUIIdentityTransfer",
             "inputs": {"model": ["modifiers", 0], "vae": ["vae", 0], "positive": positive_source,
                        "negative": negative_source, "latent": latent_source, "source_id": 2.0,
                        "phase_scale": 1.0, "id_strength": 1.0, "arcface_mode": "auto_adjust"}},
            {"id": "identity_source", "group": "Reference media", "node": "FunUILoadMedia",
             "roles": [{"at": "assets.reference_1", "input": "media_id",
                        "wireTo": {"slot": "identity", "input": "image"}}],
             "inputs": {"media_id": ""}},
        ]
        model_source, positive_source, negative_source = ["identity", 0], ["identity", 1], ["identity", 2]
    slots += [
        {"id": "sigmas", "group": "Sampling", "node": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        {"id": "sampler", "group": "Sampling", "node": "FunUISampler",
         "roles": [{"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {"model": model_source, "positive": positive_source, "negative": negative_source,
                    "latent": latent_source, "settings": ["settings", 0], "sigmas": ["sigmas", 0],
                    "seed": 0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal",
                    "denoise": 1.0}},

        *_uncrop(as_guide),
        {"id": "decode", "group": "Render", "node": "FunUIDecode", "inputs": {
            "samples": ["rejoin", 0] if as_guide else ["sampler", 0], "vae": ["vae", 0], "model": ["model", 0], "audio_vae": ["audio_vae", 0]}},
        {"id": "save", "group": "Render", "node": "FunUISaveVideo",
         "roles": [{"at": "project.video", "input": "fps", "label": "FPS", "drives": "fps"}],
         "inputs": {"images": ["decode", 0], "audio": ["decode", 1], "fps": 25.0, "filename_prefix": "FunUI"}},
    ]
    return slots


def _direct_video_to_video():
    """Plain video-to-video, no IC-LoRA: the picked video's frames are encoded and sampled again from part-way down the
    schedule (the tail of the distilled sigmas), so the result keeps the video's layout and motion and changes what
    the prompt and the noise ask for. The later the schedule starts, the more of the video survives.

    The size and length are the video's own (Video input sets them); the sound starts empty and is made with the picture.
    """
    slots = _base(False)
    at = next(i for i, x in enumerate(slots) if x["id"] == "latent")
    fps = {"at": "project.video", "input": "frame_rate", "label": "FPS", "drives": "fps"}
    slots[at:at + 1] = [
        # video_frames: the whole video is fed, not its first frame (see inputs.js).
        {"id": "source_image", "group": "Reference media", "node": "FunUILoadMedia",
         "roles": [{"at": "assets.source_image", "input": "media_id", "video_frames": True, "video_only": True}],
         "inputs": {"media_id": ""}},
        {"id": "size", "group": "Preparation", "node": "GetImageSize", "inputs": {"image": ["source_image", 1]}},
        {"id": "encode", "group": "Preparation", "node": "VAEEncode", "inputs": {"pixels": ["source_image", 1], "vae": ["vae", 0]}},
        {"id": "audio_latent", "group": "Preparation", "node": "LTXVEmptyLatentAudio",
         "roles": [fps], "inputs": {"frames_number": ["size", 2], "frame_rate": 25, "batch_size": 1, "audio_vae": ["audio_vae", 0]}},
        {"id": "latent", "group": "Preparation", "node": "LTXVConcatAVLatent",
         "inputs": {"video_latent": ["encode", 0], "audio_latent": ["audio_latent", 0]}},
    ]
    for x in slots:
        if x["id"] == "sigmas":
            x["inputs"]["sigmas"] = "0.909375, 0.725, 0.421875, 0.0"      # the last three steps of the distilled schedule: about 90% re-noised
    return slots


def _ic_lora(on):
    """The IC-LoRA of ComfyUI's own LTX 2.3 video-to-video template: the stock LoRA loader (it keeps the file's metadata)
    and the node that reads the guide's downscale factor from it. The picked video then steers through LTXVAddGuide."""
    if not on:
        return []
    return [
        {"id": "ic_lora", "group": "Loaders", "node": "LoraLoaderModelOnly",
         "inputs": {"model": ["model", 0], "lora_name": "", "strength_model": 1.0}},
        {"id": "ic_params", "group": "Loaders", "node": "GetICLoRAParameters", "inputs": {"iclora_model": ["ic_lora", 0]}},
    ]


def _uncrop(as_guide):
    """A guide sits in the latent as extra frames: take them off again before the picture is decoded."""
    if not as_guide:
        return []
    return [
        {"id": "unjoin", "group": "Render", "node": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sampler", 0]}},
        {"id": "crop", "group": "Render", "node": "LTXVCropGuides",
         "inputs": {"positive": ["identity", 1], "negative": ["identity", 2], "latent": ["unjoin", 0]}},
        {"id": "rejoin", "group": "Render", "node": "LTXVConcatAVLatent",
         "inputs": {"video_latent": ["crop", 2], "audio_latent": ["unjoin", 1]}},
    ]


#: See minimax_h3/pipeline.py: what to pass through when the node's module did not load.
OPTIONAL_NODES = {"FunUIIdentityTransfer": {"pass": {"0": "model", "1": "positive", "2": "negative"}}}


def upgrade(slots, _outputs=None):
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
    return [{"id": "ltx23_text_to_video", "title": "LTX 2.3 · Text to Video (UNTESTED)", "slots": _base(False)},
            {"id": "ltx23_image_to_video", "title": "LTX 2.3 · Image to Video (UNTESTED)", "slots": _base(True)},
            {"id": "ltx23_anchor_guide", "title": "LTX 2.3 · Anchor as guide (UNTESTED)",
             "slots": _base(True, True)},
            {"id": "ltx23_video_to_video", "title": "LTX 2.3 · Video to Video, IC-LoRA (UNTESTED)",
             "slots": _base(True, True, True)},
            {"id": "ltx23_video_to_video_direct", "title": "LTX 2.3 · Video to Video, direct (UNTESTED)",
             "slots": _direct_video_to_video()}]


PROVIDES = {"pipeline_presets": presets}
