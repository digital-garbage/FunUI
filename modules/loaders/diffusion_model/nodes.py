"""The diffusion model loader.

Picking a file and how to read it. Every option here changes what the loaded
model IS, which is why they are node widgets and not settings: nobody downstream
can supply them, and a saved workflow has to record them.
"""

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..._core import log
from .. import gguf_support, sla_attention, veda_attention
from .. import int8_convrot as int8_convrot_module
from .. import mxfp8 as mxfp8_module
from ..common import (COMPUTE_DTYPES, free_stale_copy_after_run, head_bank_metadata, WEIGHT_DTYPES, attention_choices, fp32_over_bf16, unsupported_dtypes,
                      attention_override, dtype_of, set_fp16_accumulation,
                      slow_int8_build, stored_weights, weight_model_options)


def _model_file_choices():
    """Diffusion model files, `.gguf` included.

    Core's extension set has no `.gguf`, so those files are on disk and
    invisible to every picker. Appended rather than merged in sorted order, so
    an existing pipeline's saved choice keeps its position in the list and
    nothing a user already picked moves.
    """
    return list(folder_paths.get_filename_list("diffusion_models")) + \
        gguf_support.gguf_names("diffusion_models")        # listed even with no backend: a saved pick must stay valid so the loader can say what to install



def shows(*conditions):
    """Draw an input only while each (other input, values) holds: SLA's fields belong to one method
    or engine each, and a field that does nothing in the mode picked is noise. Read by core/widgets."""
    return {"funui_shows": [{"input": name, "when": list(values)} for name, values in conditions]}

class FunUIDiffusionModelLoader(io.ComfyNode):
    # How the form lists them (core/widgets): the file, how it is stored and computed, attention, then SLA in the
    # order its choices narrow it. Not the schema's order, which saved canvas workflows read by position.
    WIDGET_ORDER = ("model_name", "weight_dtype", "int8_convrot", "mxfp8", "compute_dtype", "fp16_accumulation", "attention",
                    "sla", "sla_method", "veda_predictor", "veda_timing", "sla_engine", "sla_sparsity", "sla_sparsity_start", "sla_tau", "sla_block_size",
                    "sla_dense_steps", "sla_dense_last_steps", "sla_min_seq_len", "sla_protect_audio",
                    "sla_references", "sla_tail", "sla_stabilize_motion")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunUIDiffusionModelLoader",
            display_name="FunUI Diffusion Model Loader",
            category="FunUI/Loaders",
            description="Load a diffusion model, choosing precision and attention backend.",
            inputs=[
                io.Combo.Input("model_name", options=_model_file_choices()),
                io.Combo.Input("weight_dtype", options=WEIGHT_DTYPES, default="default",
                               tooltip="How the model is stored in memory. 'default' keeps the file as it is. A smaller type uses less memory, with a small quality cost."),
                io.Combo.Input("compute_dtype", options=COMPUTE_DTYPES, default="default",
                               tooltip="The precision used for the maths while generating. 'default' is safe; a lower one is faster, with a small quality cost."),
                io.Combo.Input("attention", options=attention_choices(), default="default",
                               tooltip="The attention method. Faster ones need extra packages and only the ones this machine can run are listed. 'default' keeps ComfyUI's own choice."),
                io.Boolean.Input("fp16_accumulation", default=False, optional=True,
                                 tooltip="Speeds up fp16 maths slightly, with a small quality trade-off. Needs a recent PyTorch and a GPU that supports it."),
                io.Boolean.Input("sla", default=False, optional=True, extra_dict={"funui_requires": ["sparse_attention"]},
                                 tooltip="Faster attention for MiniMax H3, with a quality trade-off. Needed for the SLA turbo LoRA to give any speedup. Requires Triton and an H3 model; otherwise it is skipped and the status says why."),
                # SLA settings. Every one is validated at its default (see sla_attention);
                # they are here because a knob whose value was measured is still a knob.
                io.Float.Input("sla_sparsity", extra_dict=shows(("sla", [True]), ("sla_method", ["sla", "vsa"])), default=sla_attention.SLA_DEFAULTS["sparsity_ratio"],
                              min=0.0, max=0.95, step=0.05, optional=True,
                              tooltip="How much of the attention is skipped. Higher is faster but softer. 0.90 is a good start; below 0.60 it is slower than normal attention."),
                io.Combo.Input("sla_block_size", extra_dict=shows(("sla", [True]), ("sla_method", ["sla"]), ("sla_engine", ["triton"])),
                               options=["32", "64", "128"],
                               default=str(sla_attention.SLA_DEFAULTS["block_size"]),
                               optional=True,
                               tooltip="Smaller blocks follow the audio more closely at about the same speed. Use 128 only if sound does not matter. Triton engine only."),
                io.Boolean.Input("sla_protect_audio", extra_dict=shows(("sla", [True]), ("sla_method", ["sla", "sol-attn"])),
                                 default=sla_attention.SLA_DEFAULTS["protect_audio"], optional=True,
                                 tooltip="Keeps the prompt and the sound at full attention, so the soundtrack does not degrade while the picture still looks fine. Recommended on."),
                io.Int.Input("sla_min_seq_len", extra_dict=shows(("sla", [True])),
                            default=sla_attention.SLA_DEFAULTS["min_seq_len"],
                            min=0, max=1000000, step=1024, optional=True,
                            tooltip="Short inputs below this length skip the speed-up, because it would not pay off there. Leave at the default."),
                io.Int.Input("sla_dense_last_steps", extra_dict=shows(("sla", [True]), ("sla_method", ["sla", "sol-attn", "veda"])),
                            default=sla_attention.SLA_DEFAULTS["dense_last_steps"],
                            min=0, max=8, optional=True,
                            tooltip="Runs the last N steps with normal attention for a cleaner finish. Costs extra time; 0 is fine for most runs."),
                io.Boolean.Input("int8_convrot", default=False, optional=True,
                                 tooltip="MiniMax H3 only: stores the big layers as int8 for about half the memory, with slightly different pictures. Skipped for files that are already quantized; the status says so."),
                # Added later, so after everything: saved workflows can store widget values by position.
                io.Combo.Input("sla_engine", extra_dict=shows(("sla", [True]), ("sla_method", ["sla"])), options=list(sla_attention.ENGINES),
                               default=sla_attention.SLA_DEFAULTS["engine"], optional=True,
                               tooltip="What runs the faster attention. comfy_kitchen is ComfyUI's own and needs comfy-kitchen 0.2.32+; triton is FunUI's and has more options. Falls back to triton if missing."),
                io.String.Input("sla_dense_steps", extra_dict=shows(("sla", [True]), ("sla_method", ["sla", "sol-attn", "veda"])), default=sla_attention.SLA_DEFAULTS["dense_steps"],
                                optional=True,
                                tooltip="Steps that use normal attention, counted from 1 ('1', '1-2', '1,6'). The first step sets the layout, so it is exact by default. Blank or 0 means none."),
                io.Combo.Input("sla_references", extra_dict=shows(("sla", [True]), ("sla_method", ["sla", "sol-attn"])), options=list(sla_attention.REFERENCES),
                               default=sla_attention.SLA_DEFAULTS["references"], optional=True,
                               tooltip="How reference pictures are treated. off: skipped like the video. light: keeps the most relevant parts. heavy: kept exact, but slow with big references."),
                io.Boolean.Input("sla_tail", extra_dict=shows(("sla", [True]), ("sla_method", ["sla"]), ("sla_engine", ["comfy_kitchen"])), default=sla_attention.SLA_DEFAULTS["tail"], optional=True,
                                 tooltip="comfy_kitchen engine: also keeps a rough summary of the skipped parts. Off matches the SLA turbo LoRA; untested on H3."),
                io.Boolean.Input("sla_stabilize_motion", extra_dict=shows(("sla", [True]), ("sla_method", ["sla"]), ("sla_engine", ["triton"])),
                                 default=sla_attention.SLA_DEFAULTS["stabilize_motion"], optional=True,
                                 tooltip="Triton engine: removes a faint double image on fast motion. Uses a little more memory."),
                io.Combo.Input("sla_method", extra_dict=shows(("sla", [True])), options=list(sla_attention.METHODS),
                               default=sla_attention.SLA_DEFAULTS["method"], optional=True,
                               tooltip="How the parts to skip are chosen. sla: a fixed share, what turbo LoRAs expect. sol-attn: adapts per block. vsa: only for models trained for it. veda: experimental, needs CUDA."),
                io.Float.Input("sla_tau", extra_dict=shows(("sla", [True]), ("sla_method", ["sol-attn"])), default=sla_attention.SLA_DEFAULTS["tau"], min=0.0, max=4.0,
                               step=0.05, optional=True,
                               tooltip="sol-attn only: higher skips more attention. About 1.0 is gentle, 2.0 is aggressive."),
                # Added after everything: saved workflows store widget values by position.
                io.Float.Input("sla_sparsity_start", extra_dict=shows(("sla", [True]), ("sla_method", ["sla"])), default=sla_attention.SLA_DEFAULTS["sparsity_start"],
                              min=0.0, max=0.95, step=0.05, optional=True,
                              tooltip="Skips less at the first steps and more at the last, up to the sparsity above. 0 turns it off. Try 0.70 with sparsity 0.90."),
                io.Boolean.Input("mxfp8", default=False, optional=True,
                                 tooltip="MiniMax H3 only: about twice the speed on RTX 50 and newer GPUs, with slightly different pictures. Requires torch 2.10+ with CUDA 13; otherwise skipped."),
                # Added last: saved workflows store widget values by position.
                io.Combo.Input("veda_predictor", extra_dict=shows(("sla", [True]), ("sla_method", ["veda"])), options=veda_attention.predictor_choices(),
                               default="auto", optional=True,
                               tooltip="veda: which predictor file in models/veda to use. auto picks the text-to-video one. Use the R2VA file for reference-to-video."),
                io.Boolean.Input("veda_timing", extra_dict=shows(("sla", [True]), ("sla_method", ["veda"])), default=False, optional=True,
                                 tooltip="veda: logs how long each stage took. The picture is unchanged."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, weight_dtype: str, compute_dtype: str,
                attention: str, fp16_accumulation: bool = False, sla: bool = False,
                sla_sparsity: float = None, sla_block_size: str = None,
                sla_protect_audio: bool = None, sla_min_seq_len: int = None,
                sla_dense_last_steps: int = None, int8_convrot: bool = False,
                sla_engine: str = None, sla_dense_steps: str = None, sla_references: str = None,
                sla_tail: bool = None, sla_stabilize_motion: bool = None,
                sla_method: str = None, sla_tau: float = None,
                sla_sparsity_start: float = None, mxfp8: bool = False,
                veda_predictor: str = None, veda_timing: bool = False) -> io.NodeOutput:
        asked = {k: v for k, v in locals().items() if k not in ("cls", "model_name")}          # before any local of the body
        notes = [f"FunUI Diffusion Model Loader | {model_name}"]

        accumulation = set_fp16_accumulation(fp16_accumulation)
        if fp16_accumulation and accumulation is None:
            notes.append("fp16_accumulation: unsupported by this torch build, ignored")
        elif accumulation is not None:
            notes.append(f"fp16_accumulation: {accumulation}")

        model_options = weight_model_options(weight_dtype)
        notes.append(f"weight dtype: {weight_dtype}")

        if gguf_support.is_gguf(model_name):
            path = gguf_support.gguf_path("diffusion_models", model_name)
            if not path:
                raise RuntimeError(f"{model_name} is no longer where it was listed from.")
            misnamed = False
        else:
            path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            # The CONTENT decides, not the extension. A .gguf renamed to
            # .safetensors used to reach the safetensors parser and fail with
            # a UTF-8 decode error from reading a binary header as JSON --
            # true, and no help at all.
            misnamed = gguf_support.has_gguf_magic(path)
        if gguf_support.is_gguf(model_name) or misnamed:
            if misnamed:
                notes.append(f"{model_name} is named .safetensors but is a GGUF "
                             f"container -- loaded as GGUF")
            state_dict, gguf_options, gguf_note = gguf_support.load_state_dict(path)
            # The quantized path needs its own torch operations, and they must
            # not be lost to the dtype options above -- a GGUF loaded with
            # stock ops would try to matmul block-quantized storage.
            model_options = {**model_options, **gguf_options}
            metadata = None
            notes.append(gguf_note)
        else:
            state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        converted = False                 # True once a quantizer rewrote the weights: only then is "read the file again" not the same model
        if int8_convrot:
            if weight_dtype != "default":
                notes.append(f"int8_convrot: skipped, weight dtype is forced to {weight_dtype}")
            else:
                _, note = int8_convrot_module.quantize_state_dict(state_dict, comfy.model_management.get_torch_device())
                converted = True
                notes.append(note)
                if compute_dtype != "default":
                    notes.append("int8_convrot: compute dtype is forced, so the matmuls run in that dtype -- memory is saved, speed is not")
        if mxfp8:
            mark = len(notes)
            if weight_dtype not in ("default", "bf16"):     # bf16 is what the file holds: the converted layers keep their own storage
                notes.append(f"mxfp8: skipped, weight dtype is forced to {weight_dtype}")
            elif int8_convrot:
                notes.append("mxfp8: skipped, int8_convrot already stores the weights")
            elif not mxfp8_module.available(comfy.model_management.get_torch_device()):
                notes.append("mxfp8: skipped, this machine cannot run MXFP8 matmuls (needs a Blackwell GPU, torch 2.10+ and a current comfy-kitchen)")
            else:
                _, note = mxfp8_module.quantize_state_dict(state_dict, comfy.model_management.get_torch_device())
                converted = True
                notes.append(note)
            for line in notes[mark:]:                   # the status output is rarely read; the log is
                log.info("FunUI Diffusion Model Loader", line)
        # Read before the load: comfy may consume the dict.
        gguf = gguf_support.is_gguf(model_name) or misnamed
        stored = None if gguf else stored_weights(state_dict)
        slow = slow_int8_build() if stored and "int8" in stored else None
        if slow:
            log.alert("FunUI Diffusion Model Loader", slow)
            notes.append(slow)
        if not gguf:
            metadata, heads_note = head_bank_metadata(state_dict, metadata)
            if heads_note:
                notes.append(heads_note)
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options, metadata=metadata)
        if model is None:
            # comfy returns None rather than raising, and a None reaching the
            # sampler fails much further away than the mistake.
            raise RuntimeError(
                f"Could not detect a diffusion model in {model_name}. Loading a text "
                f"encoder or a VAE file here is the usual cause.")
        if not gguf and not converted and hasattr(model, "cached_patcher_init") and model.cached_patcher_init is None and hasattr(comfy.sd, "load_diffusion_model"):
            # What LoRA training clones the model from: core's own loaders set it, and a plain file is the one case this loader can say how to read again.
            model.cached_patcher_init = (comfy.sd.load_diffusion_model, (path, model_options))

        for line in unsupported_dtypes(model, weight_dtype, compute_dtype):
            log.alert("FunUI Diffusion Model Loader", f"{line}; expect noise or broken sound. Set it to default.")
            notes.append(line)
        mixed = fp32_over_bf16(weight_dtype, compute_dtype, stored)
        if mixed:
            log.alert("FunUI Diffusion Model Loader", mixed)
            notes.append(mixed)
        dtype = dtype_of(compute_dtype)
        if dtype is not None and hasattr(model, "set_model_compute_dtype"):
            # Do NOT clear force_cast_weights afterwards. set_model_compute_dtype
            # sets it deliberately, and it is what casts each layer's WEIGHTS to
            # match. Without it the model still casts its INPUT to the requested
            # dtype (_apply_model reads manual_cast_dtype), so the first Linear
            # gets a bf16 activation against fp32 weights and sampling dies with
            # "mat1 and mat2 must have the same dtype". v4 cleared it and shipped
            # that way; it survives on constrained hardware only because the
            # low-VRAM path forces the cast back on regardless.
            model.set_model_compute_dtype(dtype)
            notes.append(f"compute dtype: {compute_dtype}")
        elif dtype is not None:
            notes.append("compute dtype: unsupported by this ComfyUI, ignored")

        override = attention_override(attention)

        # SLA wraps the chosen backend rather than replacing it: there is one
        # override slot, so the backend goes in as SLA's dense fall-through.
        # When SLA does not take (not H3, no Triton, switched off) the backend
        # is installed on its own -- asking for sparse attention must never
        # cost the backend that was picked.
        installed = False
        if sla:
            model, sla_note, installed = sla_attention.install_sla(
                model,
                sparsity_ratio=sla_sparsity, sparsity_start=sla_sparsity_start, block_size=sla_block_size,
                min_seq_len=sla_min_seq_len, dense_last_steps=sla_dense_last_steps,
                protect_audio=sla_protect_audio,
                dense_fn=override, dense_label=attention,
                engine=sla_engine, dense_steps=sla_dense_steps, references=sla_references,
                tail=sla_tail, stabilize_motion=sla_stabilize_motion, method=sla_method, tau=sla_tau, veda_predictor=veda_predictor, veda_timing=veda_timing)
            notes.append(sla_note)
        if not installed:
            if override is not None:
                model.model_options.setdefault("transformer_options", {})[
                    "optimized_attention_override"] = override
                notes.append(f"attention: {attention}")
            else:
                notes.append("attention: default (as launched)")
        else:
            notes.append(f"attention: {attention} (dense calls)")

        # Never reach into the model's shape to describe it: a log line that
        # assumes structure can fail the load it was only meant to narrate.
        kind = type(getattr(model, "model", model)).__name__
        applied_attention = ("veda" if sla_method == "veda" else "sla_h3") if installed else (
            attention if override is not None else "default (as launched)")
        # The file's own storage, not the widget: an int8 file loaded at "bf16" still runs int8.
        weights = weight_dtype if not stored else (
            stored if weight_dtype in ("default", stored) else f"{stored} in file ({weight_dtype} asked)")
        log.info("FunUI Diffusion Model Loader",
                 f"{model_name} loaded as {kind}, weights "
                 f"{weights}, compute {compute_dtype}, attention "
                 f"{applied_attention}")
        free_stale_copy_after_run(model_name, asked)
        return io.NodeOutput(model, "\n".join(notes))
