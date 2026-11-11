"""MiniMax H3-Image support: text to image, image editing and multi-picture references.

H3-Image is the still-picture sibling of the H3 video model. It shares the text encoder and
the 16x latent grid, has no sound stream and reads at most 16 pictures: the first one is the
picture to edit (`<Picture 1>` in the prompt), the others are references (`<Picture 2>`, ...).
It samples in 6 steps at CFG 1 and works best around 4 to 16 megapixels.

The result is a picture file (PNG), not a clip. The pipeline offers no Length or FPS; the preset sets
the project's Length and FPS to 1, so a scene made with it counts as one second on the timeline.

The checkpoint is recognised by its header: the packed-DiT keys of H3 with no audio projection.
Everything the model needs at run time (the image latent, the picture labels, the sigma grid
shipped inside the file) is ComfyUI's own.
"""

from ..._core import traits as _traits

ID = "model_minimax_h3_image"
TITLE = "MiniMax H3-Image"
STAGE = "load"
CATEGORY = "system"
STATUS = "experimental"

# Same key-name signature ComfyUI detects this architecture by (comfy/model_detection.py): the
# image checkpoints have `condition_proj` and no `audio_patch_proj`. Bare, and the full-checkpoint prefix.
_PREFIXES = ("", "model.diffusion_model.")
_PRESENT = ("video_patch_proj.weight", "condition_proj.weight")
_ABSENT = "audio_patch_proj.weight"

#: How many pictures the model reads (the one to edit and its references).
MAX_REFERENCES = 16

#: The project numbers the preset sets: one frame at one frame per second.
PROJECT = {"num_frames_per_scene": 1, "frame_rate": 1}


def detect(keys) -> bool:
    keyset = set(keys)
    return any(all(f"{p}{k}" in keyset for k in _PRESENT) and f"{p}{_ABSENT}" not in keyset for p in _PREFIXES)


def probe_traits(keys) -> list:
    """What core reads off the loaded model (a 3-axis latent with a length-1 time axis, flow prediction), said before it loads."""
    if not detect(keys):
        return []
    from comfy.latent_formats import MiniMaxH3Video
    found = ["predict_flow"]
    rank = _traits.LATENT_RANK.get(MiniMaxH3Video.latent_dimensions)
    if rank:
        found.append(rank)
    if getattr(MiniMaxH3Video, "temporal_downscale_ratio", 1) not in (None, 1):
        found.append("temporal_compression")
    return found


def text_to_image():
    slots = [
        {"id": "model", "group": "Loaders", "node": "FunUIDiffusionModelLoader", "inputs": {
            "weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
        {"id": "clip", "group": "Loaders", "node": "FunUICLIPLoader", "inputs": {"type": "minimax"}},
        {"id": "vae", "group": "Loaders", "node": "FunUIVAELoader", "inputs": {"dtype": "bfloat16"}},

        # The picture only supplies the SHAPE (and none at all means the project's own); the scale is the
        # project's width and height, snapped to the 32-pixel grid the model wants.
        {"id": "source_image", "group": "Reference media", "node": "FunUILoadMedia",
         "roles": [{"at": "assets.source_image", "input": "media_id",
                    "blank_like": {"slot": "aspect", "width": "width", "height": "height"}}],
         "inputs": {"media_id": "", "blank_width": 0, "blank_height": 0}},
        {"id": "aspect", "group": "Reference media", "node": "FunUIAspectSize",
         "roles": [{"at": "project.video", "input": "width", "label": "Width", "was": "target_width"},
                   {"at": "project.video", "input": "height", "label": "Height", "was": "target_height"},
                   {"at": "project.resolution_basis", "input": "basis"}],
         "inputs": {"image": ["source_image", 0], "width": 2048, "height": 2048, "multiple_of": 32, "basis": "area"}},

        # With no reference picture this is text to image; the first one is the picture to edit.
        {"id": "cond", "group": "Preparation", "node": "MiniMaxH3ImageConditioning",
         "roles": [{"at": "generation.prompt", "input": "prompt", "label": "Prompt"}],
         "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": "", "width": ["aspect", 0], "height": ["aspect", 1]}},
        # The model takes no negative prompt; the sampler node only requires some conditioning here.
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "project.negative", "input": "text", "label": "Negative prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "settings", "group": "Preparation", "node": "FunUIModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunUILoadModifiers",
         "inputs": {"model": ["model", 0], "settings": ["settings", 0]}},

        {"id": "sampler", "group": "Sampling", "node": "FunUISampler",
         "roles": [{"at": "generation.sampling", "input": "steps", "label": "Steps"},
                   {"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.sampling", "input": "scheduler", "label": "Scheduler"},
                   {"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {"model": ["modifiers", 0], "positive": ["cond", 0], "negative": ["negative", 0],
                    "latent": ["cond", 1], "settings": ["settings", 0],
                    "seed": 0, "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},

        {"id": "decode", "group": "Render", "node": "FunUIDecode",
         "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0], "model": ["model", 0]}},
        {"id": "save", "group": "Render", "node": "SaveImage",
         "inputs": {"images": ["decode", 0], "filename_prefix": "FunUI"}},
    ]
    for n in range(1, MAX_REFERENCES + 1):
        # Not wired here: an unused picture input is simply absent. The app wires the ones a scene has, per run.
        slots.append({
            "id": f"ref_source_{n}", "group": "Reference media", "node": "FunUILoadMedia",
            "roles": [{"at": f"assets.reference_{n}", "input": "media_id",
                       "wireTo": {"slot": "cond", "input": f"images.image_{n}"}}],
            "inputs": {"media_id": ""}})
    return slots


def presets():
    return [{"id": "minimax_h3_image", "title": "MiniMax H3-Image · Text to Image and Edit",
             "slots": text_to_image(), "project": PROJECT}]


PROVIDES = {"detect": detect, "probe_traits": probe_traits, "pipeline_presets": presets}
