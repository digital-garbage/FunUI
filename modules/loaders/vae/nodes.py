"""Loading a VAE, with an explicit dtype and a report of what happened.

Two differences from core's VAELoader, both carried over from v4 where they were
proven: the dtype is chosen rather than guessed, and a second output narrates
what actually loaded. Silent success that produced the wrong thing is the failure
mode these exist to remove.
"""

import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

from ..._core import log

# Chosen, not inferred. bf16 is the default because fp16 is a known failure on
# the AV VAEs, and fp32 decodes slowly for no visible gain.
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def vae_files():
    # Empty is a legitimate state -- a fresh install has no models. Returning an
    # empty list lets the node exist and say so, rather than failing to register.
    return folder_paths.get_filename_list("vae")


class FunUIVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            # Frozen once shipped: this string is stored as `class_type` in every
            # saved workflow, so changing it silently breaks them.
            node_id="FunUIVAELoader",
            display_name="FunUI VAE Loader",
            category="FunUI/Loaders",
            description="Load a VAE with an explicit compute dtype.",
            inputs=[
                io.Combo.Input("vae_name", options=vae_files()),
                io.Combo.Input("dtype", options=list(DTYPES), default="bfloat16",
                               tooltip="The precision the VAE works in. bfloat16 is the default; MiniMax H3 needs it."),
                io.Combo.Input("decode_vae_name", options=["None", *vae_files()], default="None", optional=True,
                               display_name="Decode-only VAE (Optional)",
                               tooltip="A second VAE used only for the final picture decode; this one still encodes "
                                       "(references, start pictures). For a decoder that has no encoder: MiniMax H3's "
                                       "Sol-H3-VAE (sol_h3_vae.safetensors, whole clip at once, about 2.5 times faster; "
                                       "needs a ComfyUI that supports it) or an X2 Detail VAE. None decodes with this VAE."),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, vae_name: str, dtype: str, decode_vae_name: str = "None") -> io.NodeOutput:
        vae = cls._load(vae_name, dtype)
        status = f"{vae_name} loaded as {dtype}"
        if decode_vae_name and decode_vae_name != "None":
            # Rides on the VAE object: the decode node prefers it, everything else (encoding) uses this VAE.
            try:
                vae.funui_decode_vae = cls._load(decode_vae_name, dtype)
                status += f"; {decode_vae_name} decodes the finished picture"
            except Exception as exc:            # noqa: BLE001 -- a decoder this ComfyUI cannot read must not stop the run
                log.alert("FunUI VAE Loader", f"{decode_vae_name} could not be loaded ({exc}). The main VAE decodes instead. "
                          "A ComfyUI that supports this VAE is needed to use it.")
                status += f"; {decode_vae_name} could not be loaded, {vae_name} decodes"
        return io.NodeOutput(vae, status)

    @staticmethod
    def _load(vae_name: str, dtype: str):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        # An audio VAE's own file keeps its model config in the header, and may name its parts as a whole checkpoint does.
        state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"audio_vae.": "autoencoder."})
        vae = comfy.sd.VAE(sd=state_dict, dtype=DTYPES[dtype], metadata=metadata)
        if vae.first_stage_model is None:
            keys = sorted(state_dict)
            raise RuntimeError(f"ComfyUI did not recognise {vae_name} as a VAE ({len(keys)} weights, first keys: "
                               f"{', '.join(keys[:6])}; header: {sorted(metadata or {})[:6]})")
        log.info("FunUI VAE Loader", f"{vae_name} loaded as {dtype}")
        return vae
