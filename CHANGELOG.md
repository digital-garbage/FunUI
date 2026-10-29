# Changelog

## [Unreleased]

### Added

- **Two stages: Latent upscaler as an option** (Settings > Engine > Sampling, "How the hand-over is enlarged"). Bicubic stays the default; plain choices Lanczos, Bislerp, Bilinear, Trilinear (same as Bilinear here, time is never resampled), Area and Nearest-exact are there too (Lanczos is done on the numbers, not through 8-bit pictures as ComfyUI's would, which a latent cannot survive). Or pick Latent upscaler and a file from models/latent_upscale_models (H3's convolution upscaler) to enlarge the first stage's clean guess with the learned network instead. It runs at the ratio that reaches the full size (it works in even steps, then a bicubic trim fits the exact grid), once per run, on a small latent; the file is loaded once and kept. A missing or unreadable file falls back to bicubic with a note. Untested on a GPU.
- **Upscale no longer makes the next generation reload the models.** It ran as an extra ComfyUI prompt without the workflow's loaders, and ComfyUI's cache lets go of loaders a prompt does not use; it now runs as a background job of FunUI, beside ComfyUI's queue (Interrupt in ComfyUI no longer stops it).
- **Interpolate** (module of its own, add-on **NVIDIA RTX Frame Generation**): adds in-between frames to a finished render with NVIDIA's RTX frame generation (2 to 4 times the frames; the length and the sound stay, 24 fps becomes 48 fps). Settings > Editor > Interpolate: Never, a button beside the rating, or always when a render finishes. Quality Fast, Balanced or Best, and 10-bit processing. It runs as a background job of FunUI, not as a ComfyUI prompt, so it does not make ComfyUI drop the workflow's loaded models (an extra prompt without loaders does exactly that, and the next generation reloads them); the NVIDIA model also stays loaded between presses. Works beside Upscale in either order and never runs twice on the same file. Needs an RTX card and nvidia-vfx 0.2 or newer: switching the add-on on moves an older nvidia-vfx up (nothing else is touched), and Super Resolution never moves a newer one back. ComfyUI's own RIFE/FILM interpolation nodes are a different thing and are not used. Works on a rental GPU (a few seconds for a clip).
- **Project settings > Resolution is the** (H3, LTX image to video): what Width and Height mean once the scene picture sets the shape. Short side (the smaller number is the clip's shorter side), Long side (the larger is its longer side) or Pixel count (the old rule: about Width x Height pixels, which turned 1024x1024 into 1312x768 for a wide picture). New projects start on Short side; projects saved earlier keep Pixel count, so nothing changes size by itself. The H3 preset no longer needs the KJNodes ImageTransformKJ node for this (saved H3 pipelines are switched over on load).
- **VAE Loader: Decode-only VAE (Optional)**: a second VAE used only for the final picture decode while the first still encodes (references, start pictures). For decoders without an encoder: MiniMax H3's Sol-H3-VAE (`sol_h3_vae.safetensors`, whole clip at once, about 2.5x faster decode, needs a ComfyUI newer than 0.43.1) and the X2 Detail VAE.
- **H3 loader: MXFP8** (Blackwell GPUs): stores the attention and MLP weights of the middle blocks as MXFP8 (first 2 and last 3 blocks stay bf16) for about twice the matmul rate. Needs torch 2.10+ with CUDA 13; skipped with a note elsewhere. Untested on a GPU; listen to the sound first.
- **Two stages** (Settings > Engine > Sampling, H3): the first steps of a run go on a smaller picture (First stage size, 0.7 by default), the model's clean guess is resized to the full size and the last steps finish there. Works with any sampler; skipped for a run that starts from a picture or a mask. Untested on a GPU.

## [5.1.0] "Classy Chopper" - 2026-10-27

### Changed

- **LTX users: you no longer drop pictures on the timeline to feed the model.** The start picture is chosen in the scene's References selector (reference picture 1), and Image to Video and Video to Video are separate workflows: switch the preset (Settings > Models & Pipeline) to change between them; a pipeline is not interchangeable between the two.
- **On LTX only Noise merging, the Attention explorer and Taste are offered for now.** Every other sampling and guidance enhancement (REINS, Block repeat, Identity bank, Guide frames, Shot memory, Camera move, Temporal style and the rest) is held back there until it has been proven on a real LTX run; H3 is unchanged. A model holds a module back with a `hides:<module id>` trait, so taking one off the list in `modules/models/ltx/__init__.py` (`HELD_BACK`) is all it takes to bring it back. Saved settings are kept; the Log says "held back for now".
- **Load Media decodes only what it keeps**: a prepared video decodes just the kept frames and shapes each as it comes, and a picture input reads one frame, so a long clip no longer sits whole in memory.
- **A clip named in Japanese (or any non-ASCII name) no longer fails a run on a machine whose console is not UTF-8.**
- **Restoring a saved pipeline no longer adds another preset's input slots** (an Image to Video loader landing in a Video to Video pipeline made a picture reference fail with "identity: there is no such slot").
- **The ComfyUI log survives a restart**: the console is kept on disk as it prints, and the Log window has a "Previous run" button for the run before the last start. The Log also says how much RAM and GPU memory is in use when a LoRA is applied and when sampling starts, so a run the box kills for memory leaves a trace.
- **Shot camera and SLA are H3-only**: models announce `shot_prompts` and `sparse_attention` (only H3 does), shot camera requires the first and the loader's SLA fields the second, so both are hidden when an LTX model is picked.

### Fixed

- **An older LTX Image to Video pipeline now falls back to text to video** when no picture is fed in (its start slot is upgraded on load to bypass the anchor and use a plain stand-in), instead of refusing to start. The fallback chain is now also run end to end in a test.
- **Cancelling during a Noise merging second pass now cancels** instead of being swallowed as a failed merge.
- **Identity bank's lock** no longer counts probe passes (such as Noise merging's second answer) as steps of the run.
- **Attention explorer ratings made on H3 no longer teach LTX** under the same Taste key: the two model families keep their ratings apart.
- **Guide frames** refuses a NaN position instead of crashing.

### Added

- **Block settings know the model**: the file's header gives its block count, the Blocks hint says "This model has blocks 0-47", and a list is cut to those blocks when you commit it.
- **Temporal style: auto** picks a style for each shot from its own words (motion words punch, calm words hold, loop words loop).
- **Randomizer, REINS and Block repeat on LTX** (new traits `block_scope` and `block_rows`); the other block-hook features stay H3-only. Whole-span repeat runs as each-block on LTX.
- **Attention explorer on LTX**: the depth groups and the clip regions (four time bands of the picture, and the sound) work; Prompt focus stays H3-only and is left out on LTX.
- **Identity bank** (LTX, needs a Taste key): keeps the attention look of the clips you like and leans on it in the next ones of the same size (BachVid), harder on steps where the picture is still changing (KV-Lock). Untested on a real LTX run.
- **Guide frames** (LTX, Settings > Engine): lays the middle picture of the clip before, or one picture from each of the last clips (needs the JoyAI-Echo LoRA), into the shot as weak guides through ComfyUI's own LTX guide attention. The Image to Video preset has the new slot and crops the guides again before decoding.
- **LTX 2.3 is tested** on a real model, with two presets: **Video to Video, direct** (the video is encoded and sampled again from part-way down the schedule; the sigmas slot, default `0.909375, 0.725, 0.421875, 0.0`, says how much changes) and **Image to Video, from References** (reference picture 1 is the start frame; nothing is picked twice). The earlier LTX presets (Text to Video, Image to Video, Anchor as guide, IC-LoRA) are gone. **Text to video is the fallback in both**: with no picture or video fed in, the anchor is bypassed (image to video) or a grey clip of the project's size and length is sampled with the whole schedule (video to video), and the run says so. Presets load separate files (diffusion model, video VAE, audio VAE, text encoder). LoRAs and Noise merging work on LTX; other enhancements are not tested there yet. Video to Video shows Width, Height and Length "(no video)" in Project settings for that fallback.
- **LTX Image to Video takes its shape from the aspect source**: the picture on the timeline gives the clip its aspect ratio and the project's Width x Height give the pixel budget (new **FunUI Aspect Size** node), as on H3. With no such picture the project's own shape is used. It is not the start frame: that is reference picture 1.
- **Flip width and height** button in Project settings, and on the Video input card.
- **Video input** (Project and Scene tabs): reference videos (and, on LTX, a video picked as the scene start, fed as frames) from the Media bin can be got ready before the model sees them, like a video loader node: target width and height (fill, fit with bars, or stretch; rounded to a multiple), edge crop in percent, frame rate (project's, the video's own or a number), skip first N frames, cut last N, every Nth frame, frame cap. The sound is cut to match. One set per project and one per scene; a dropdown in Project settings says whose is used (Automatic: the project's when switched on, otherwise the scene's). Off by default, so existing projects load videos as before. Works for any model whose pipeline takes reference videos (MiniMax H3 today). Tested on made-up frames and an encoded file, not yet on a rental.
- **Attention explorer** (needs a Taste key): each run tries a small attention change (sharper, as is or looser) per depth group of blocks, and your ratings decide which changes stay; how far a change goes learns from ratings too (or type your own). The Log names the exact blocks made sharper or looser. Also learns, per run: how strongly each group attends to your prompt ("Prompt focus"), and the clip's four time bands and its sound separately ("Clip regions"). A rating counts more for runs of a similar prompt, and a combination you liked on this prompt is reused (with some groups re-drawn). Zero extra model passes. Verified on H3 and LTX.
- **Noise merging** (Settings > Engine > Sampling): "Merge" can also run a whole second video beside yours from the second seed ("Second run": one extra pass at every step up to the last one named, blended at the steps you name), or mix the run's starting noise with a second seed's (no extra cost, variance kept). Or, at the steps you name, the step is answered a second time on the same picture with different noise (a second seed), and the two pictures are blended at your strength. Merge order picks which one is the base. The second seed is random each run unless you type one. Picture and sound are both merged (the sound can be left out with "Merge sound too"). "Second pass new noise" sets how much of the step's own noise the second pass swaps for the second seed's (1 all, lower keeps more of your own). Costs one extra model pass per merged step. Verified on H3 and LTX.
- **SLA rising sparsity** (diffusion model loader, `sla_sparsity_start`): skip fewer attention blocks on early steps and more on late ones, in a straight line up to `sla_sparsity`. Off at 0. One real H3 render (4 steps, 0.70 to 0.90): quality held, perhaps slightly better. Idea from GSRS (arXiv 2610.21129); only the schedule is taken, not its other parts.

## [5.0.3] "Classy Chopper" - 2026-10-25

### Changed

- **FunPack is now FunUI.** It outgrew "pack": it is a full interface. New logo, a wink `;)`, before the name in the top-left corner and in Settings > About. Node ids, log lines and docs follow the new name (`FunUISampler`, `FunUI/Loaders` and so on). The 5.x codename is now "Classy Chopper".
- **The app is served at `/fun/`** instead of `/funpack/`. The old address redirects.
- Picker wheel settings are saved with the project.
- Repository moved to `digital-garbage/FunUI`; GitHub redirects the old address, so existing clones keep updating.

### Added (experimental)

- **PTD few-step** (Settings > Engine > Sampling, MiniMax H3 only): renders in the 4 to 8 hops of a PTD adapter file from `ComfyUI/models/ptd` instead of a long schedule, with no CFG. Reads the EPFL fl2va and taobao-content-ai Ref2VA releases, as bf16 or int8, on the full or pruned base; ordinary LoRAs stack with it. The chosen sampler, steps and scheduler are replaced while it is on. Ported from ComfyUI-PTD-Compact (Nynxz). Runs on a real H3 (Ref2VA, 4 hops, int8); best with a turbo-merged base, SLA off and enhancements off.

### Changed in Decisiveness

- The manual setting is now a slider "Commitment" from -10 to +10 (0 is off). A value saved by an older version resets to 0.

### Migrated on load

- Projects, imported workflows and stored settings that carry old `FunPack…` names are renamed as they are read. Browser settings (wheel, welcome, timeline peek, model list) are carried over once. Nodes already placed in ComfyUI's own graph under the old `FunPack…` ids need adding again.

## [5.0.2] - 2026-10-22

### Changed

- **Interface** follows Apple's Human Interface Guidelines more closely: roomier windows and gaps, text contrast of at least 4.5:1, nothing under 11px, a solid focus ring, a check mark and bolder label on selected choices, and a slider value you can type. Composer and Training open and close with a Genie motion; the preview picture scales to fit its stage.
- README: new screenshots of the redesign and a clearer note on how the project is maintained.

### Fixed

- **Restart ComfyUI** no longer shows "Restarting ComfyUI…" twice while it waits.

## [5.0.1] - 2026-10-21

Fixes from real H3 runs on rentals, plus the first usable LoRA training.

### Fixed

- **Save Video** no longer loses a video when the GPU encoder will not open: it saves on the CPU and the Log says why. NaN/Inf sound is dropped with a warning and the video is kept.
- **Reference videos and sound** (H3 Reference to Video) go to R2V's own video and audio inputs, are no longer cut to 15 seconds, and a video's soundtrack comes along. Verified on H3. Old projects get the new slots on open.
- **Render** button was dark with near-black text.
- **RTX VSR** says why a load failed (missing `libnvidia-ngx` on rentals).
- **Taste direction** works on the CPU (no more cuda/cpu mismatch in the prompt nudge and score slider).
- **Text-only H3 runs** no longer need the preset's `blank_like`.
- The Log now survives a crash, and notes a project whose length changed on save.
- Media upload shows percent and no longer stalls the server on big clips.
- Shot Camera no longer re-encodes the H3 prompt on every run.

### Added

- **Log colours**: a healthy log is all plain text. Only lines that need a look are coloured: warnings and alerts in amber, errors and tracebacks in red, in the Log window and in the terminal.
- **LoRA training** now runs on H3 with FunPack's own loop (ComfyUI stays the backend): block swap, previews at chosen steps, Pause and Resume, Restore and continue, copies saved every N steps, live loss and time graphs, downloads, encoded data reused between runs. Training window is a step-by-step wizard (Data, Tags, Train, Results). Verified on H3; other models still use ComfyUI's trainer and are untested.
- **Gallery**: saved videos and pictures next to Temp files, with preview, Select mode and a confirmed delete.
- **15 reference slots for H3** (9 pictures, 3 videos, 3 sounds); FunPack Load Media takes a picture, video or sound.
- **Models & Pipeline**: a bypass toggle per node. Quick wheel: Disable all enhancements, Fix the seed.
- **FunPack Timing** splits the time before step 1 into model preparation, setup and the first forward pass.
- Loading window names the module being loaded.

### Added (experimental)

- **Distilled Flow samplers** (experimental, untested on a real model): plain names in every ComfyUI sampler list, no settings. `dflow_ord1` is Euler; `dflow_ord2` reuses the last prediction for second order (no extra model calls); `dflow_ord2_ab2` fades the second order in across the run. Sound riding in the latent stays first order. Back from 4.x; its extras (velocity bias, ALG, momentum, sharpness) are separate modules now.
- **Sharpness** (experimental): wins back fine detail in the last part of the run with any sampler that calls the model once per step; no extra model calls. Settings > Engine > Sampling. Not the spatial latent sharpening that was tried and removed in 4.x.

## [5.0.0] "Cunning Cranberry" - 2026-10-16

A rewrite from scratch, not an update of 4.x. Same idea (a studio for video inside ComfyUI that learns from your ratings), new architecture: every feature is a module that announces itself, so a feature cannot be forgotten in the UI, a wire cannot be missed, and a failing optional feature module is left out while the run carries on. 4.x projects are not imported; see "Not carried over".

### What it is

- **A browser studio at `/funpack/`**: timeline with one clip per scene, Composer (story in one box, cut word, templates, `$variables`, shortcuts), properties, media bin, preview, takes, rating, Export, Render, project wizard and guided tour.
- **Models & Pipeline**: the pipeline as cards. Pick files, swap or add nodes (any installed custom node), wire one node's output into another's input, import a ComfyUI workflow. The server refuses what would not build and says why.
- **Modules** (Settings ▸ Modules): each feature switches on or off on its own. A pipeline runs without an optional node whose module did not load, and says so. A module that fails during a run is left out of it, the log says why, and it stays off until its code changes and ComfyUI restarts, or you turn it back on.
- **Learning from ratings**: a Taste key remembers ratings; the guidance, steering and prompt modules that use them start acting once they have enough ratings (a few liked clips for some, ten or more rated clips for others; some also need disliked ones) and say so when they need more.

### Models

- **MiniMax H3**: the model v5 is built and tested around. Presets for reference to video; works text-only (no source picture). A second-pass preset exists in the code but is not offered until it has been verified.
- **Stable Diffusion 1.5**: experimental.
- **LTX 2.x**: **UNTESTED.** Never run on a real LTX model, and nobody plans to test it soon. Presets and the module are marked UNTESTED, and LTX runs log the same notice. You are free to try it; please report problems in Issues.

### Features

- **Proven**: prompt markup, empty latent, second pass (the H3 preset for it is held back), loaders (diffusion model, CLIP, VAE, LoRA, checkpoint, media), cut opening, decode, save video, anchor de-staticking (ALG), randomizer, DynaShift, first-step seed search, modifiers, sampler, detail, shot camera, prompt enhancer, Render, taste modules (slider, guidance, prompt nudge, attention, REINS, early guidance), shot memory, decisiveness, late-branch guidance, momentum, velocity bias.
- **Experimental**: continuity, identity transfer, negative erase, camera move, context windows, block influence, block repeat, phrase probe, shadow negative, video detail, temporal style, massive-activation steering, video-to-audio coupling, **LoRA training**.
- **Prompt enhancer**: a language model rewrites each finished prompt. Has its own encoder pick (Composer ▸ Enhance ▸ Encoder), because H3's own encoder cannot write prose. Verified with Gemma 4.
- **Auto Montage**: builds a new cut at the end of the timeline from your rendered clips; originals stay untouched, and the pieces are never regenerated or rated.
- **Upscale** (a background job of FunUI like Interpolate, so it no longer makes ComfyUI drop the workflow's loaded models) a finished render with an upscale model (`.safetensors`) or NVIDIA RTX Super Resolution.
- **LoRA training** (add-on, Settings > Modules): a Training window with four steps. Data (optional): drop files, pick from your Bin, or take your rated results (Liked, Disliked, Bad composition, Bad image, All). Tags: a caption editor beside every picture, a tag cloud, bulk edits and automatic tagging with downloadable tagger models. Train: all the settings, a live loss graph, Pause and Resume, and previews: every N steps training waits while your own pipeline makes a picture or clip with the LoRA so far, then carries on. Results: past runs. It trains everything in the set. H3 (picture and sound together) uses the module's own training loop with ComfyUI carrying the batches; other models use ComfyUI's trainer, with the loaders in your pipeline. **Experimental, and UNTESTED on real models or GPUs**: proven only on tiny stand-in models on CPU.
- **Add-ons** (spaCy for automatic camera and focus, nvidia-vfx, onnxruntime for LoRA training) install on the next start after you switch them on.

### Not carried over

- 4.x projects and saved settings are not imported: start new projects in 5. ComfyUI workflow files can still be imported in Models & Pipeline.
- Features that never proved out stay out: plateau step-cache, block steering, NVFP4 loader.
- The unwired Engine Appearance module (the theme lives in Settings ▸ Appearance). Settings saved while it existed are ignored, not refused.

### Known limits

- **LTX 2.x is untested** (see Models).
- **`.gguf` text encoders** need the ComfyUI-GGUF node pack and work only for architectures it understands; otherwise they are refused with a message. Gemma 4 `.gguf` does not work; a loader for it may come later.
- **H3 start frame**: in the H3 preset the scene picture sets the video's shape only; a start frame is not part of it.
- **H3 second pass** is held back: not offered until verified.
- The ready-made pipelines need Save video, Load media, the loaders, the sampler and decode: if the module behind one of those fails to load, Generate names the missing node and refuses. The optional nodes (Enhance prompt, Shot camera, Prompt markup, Negative erase, Cut opening, Identity transfer) are skipped instead, with a note beside Generate, and their controls are hidden. Which nodes are optional is declared by the model module (MiniMax H3, LTX), so it applies while that module loads. The one exception: the Reference 1 picture of Identity transfer stays on screen and does nothing until the module is back.
- Turning some modules on switches the model compiler off for the run, which costs sampling time.

### Docs

- [How to start generating with FunPack](docs/how-to-start.md).
