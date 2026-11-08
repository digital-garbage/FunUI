# How to start generating with FunUI

This is the short path from "installed" to "first clip". It does not explain every node or loader: the app does that where you use them (hover a control, read its hint).

## 1. Install

With git, inside `ComfyUI/custom_nodes`:

`git clone https://github.com/digital-garbage/FunUI`

For the classic 4.x add `-b v4-classic`. `comfy node install FunUI` and ComfyUI-Manager install what is published on Comfy Registry, which is 5.0 only once it is published there; check Settings ▸ About FunUI after installing.

Then `pip install -r requirements.txt` in the same Python environment ComfyUI uses, and restart ComfyUI. FunUI uses your existing ComfyUI and PyTorch. If pip wants to upgrade a package ComfyUI compiles against (torch, numpy, transformers, comfy-kitchen), read what it will change first: FunUI's own updater only installs what is missing and reports the rest.

## 2. Open FunUI

Start ComfyUI as usual and open `/fun/` on the same address, for example `http://127.0.0.1:8188/fun/`. The chip at the top right says whether ComfyUI is live.

## 3. Make a project

**File ▸ Project Setup Wizard** has three steps (Back and Continue sit in its footer): choose your look, name the project and pick **Start from** (the pipeline of the model you will use), then say yes to the tour if you want a walk through the screen (Help ▸ Start tour runs it again).

| Start from | State |
|---|---|
| MiniMax H3 · Reference to Video, MiniMax H3 · Fast & Simple R2V | The model FunUI is built and tested around. |
| Stable Diffusion 1.5 · Text to Image (held as a clip) | Experimental. |
| LTX 2.3 · Image to Video, from References, LTX 2.3 · Video to Video, direct | Tested on a real model, with LoRAs and Noise merging. Other enhancements are not tested on LTX yet; please report problems in [Issues](https://github.com/digital-garbage/FunUI/issues). |

## 4. Pick your files

The model's pipeline starts empty: it needs your files.

1. Open **Settings** (in the menu bar, next to Help) and then **Ready to generate?**. It checks the machine: ffmpeg, GPU, model folders, node packs.
2. Open **Settings ▸ Models & Pipeline** and choose a file in each card marked with an orange dot (diffusion model, text encoder, VAE, and so on). The page says how many cards still need something and says "Every slot is filled" when you are ready.

Put model files where ComfyUI expects them (`models/diffusion_models`, `models/text_encoders`, `models/vae`, `models/loras`). 

`.gguf` text encoders need the ComfyUI-GGUF node pack (FunUI does not install it) and work only for architectures it understands; otherwise they are refused with a message (Gemma 4 `.gguf` is one such case). Use the `.safetensors` file of the same model. A diffusion-model `.gguf` also loads without the pack, but is expanded to full size in memory.

## 5. Write the story

Click **◆ Composer** (right end of the timeline header) and write all scenes in one box, separated by the cut word (see the Cuts tab). Each piece becomes one clip on the timeline. Composer also keeps `$variables`, templates and shortcuts, and has an optional **Enhance** tab: a language model rewrites each finished prompt. On H3, pick a chat encoder (for example Gemma 4) under Enhance ▸ Encoder, because H3's own encoder cannot write prose. Leave Encoder family at its default unless your encoder belongs to another family (Gemma 4 worked with the default).

## 6. Generate

- **▶ Generate** (timeline header) makes every scene on the timeline (not left-out scenes or video clips). An earlier render of a scene stays as a take.
- **Selected** makes only the picked scene. **Takes ×N…** makes the same scene several times, differing only by seed.
- **⏱ Sampler** (right end of the same header) holds steps, sampler, scheduler, seed and the common switches. While a run goes, the step count shows to its left.
- No picture needed: on H3, a scene with no source image is generated from text alone.

Watch the Log window if nothing happens: any module that fails says why there, and the rest of the run goes on without it.

## 7. Cut, render, export

Trim and split clips on the timeline. The **＋ Add** menu in the timeline toolbar adds a scene, a video clip or audio track from the bin, an effect or transition, or a text or picture overlay. **⧉ Render** (timeline header) joins the timeline into one file; **⤓ Export** (toolbar) downloads the picked clip (picture only: a clip whose sound was separated is silent there, Render mixes it).

## 8. Teach it your taste (optional)

Rate a result with **Rate scene…** in the timeline header, next to **Takes ×N…** (liked, disliked, bad image, bad composition). **Upscale** and **Interpolate** appear beside it once you set them to "When I press" in their settings. Set a **Taste key** (the ⚠ chip on the timeline points to it: **Settings ▸ Engine ▸ System**) so ratings are remembered. **Settings ▸ Refinement & Taste** lists, exports and deletes keys. Modules that learn from ratings start acting once they have enough ratings (a few liked clips for some, ten or more rated clips for others; some also need disliked ones) and say so when they need more.

## 9. Features marked experimental

**Settings ▸ Modules** switches each feature on or off. Several features are experimental: built and run, but not yet proven over many runs. If a module fails while generating, the run carries on without it, the Log says why, and the module stays off until its code changes and ComfyUI restarts, or you turn it back on in Settings ▸ Modules.

## If something goes wrong

- Nothing happens: check the ComfyUI chip and the Log window.
- Wrong look: Settings ▸ Appearance. Lost a layout: View ▸ Reset Layout. Lost a project: File ▸ Open recent.
- A bug or a missing feature: please open an [issue](https://github.com/digital-garbage/FunUI/issues).

## Known limits in 5.0

- LTX 2.3 is tested for the two presets above; other enhancements are not (see above).
- In the H3 preset the scene picture sets the video's shape only; a start frame is not part of it.
- The H3 "second pass" preset is not offered yet: it has not been verified.
