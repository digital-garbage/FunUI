# Changelog

What is new in each version, in plain words. Newest first.

## [Unreleased]

- **MiniMax H3-Image** (experimental): a new "Start from" pipeline, MiniMax H3-Image · Text to Image and Edit. Write a prompt for a picture, or add reference pictures and refer to them as `<Picture 1>`, `<Picture 2>`, ... in the prompt (the first is the picture to edit; up to 16). The result is a PNG picture, not a clip: the preview shows it as a picture, and Length and FPS are not offered for it (both are set to 1). REINS and the Attention explorer work on it (the explorer leaves out the sound and clip regions, which it does not have); Seed explorer and Decisiveness are offered too. You need the model, its VAE and its text encoder files; the model's licence excludes the EU, the UK, South Korea and the United States.
- **Noise merging is now called Step merging**, because most of its modes merge the model's answers at a step, not noise. Settings are kept.
- **Alternating prompts** (Settings > general): write `[dance|zoom]` in a prompt, or `DANCE|ZOOM` between two shortcuts, for two options, A and B. "Main prompt reads" sets what the prompt says by default: "A while B" (default), "A B", only A or only B. Switch on "Read A and B separately" (off by default) to also assign steps: "Steps that read only A" and "Steps that read only B" give each step you name its own option, with no extra model pass. A `|` next to anything that is not a shortcut is left as typed. Two options per bracket.
- **Simultaneous actions** (Step merging, needs "Read A and B separately"): with `[A|B]` in the prompt, each merged step is answered once on A and once on B from the same picture, and the two answers are mixed at your strength. "Second pass new noise" sets how much of the noise B starts from is new (0 keeps the same noise for both). Strength 0 reads only A, and strength 1 with new noise 1 reads only B, each with a single pass. Steps named in Step merging win over the per-step lists. Costs one extra model pass per merged step.
- **Removed: timed phrases `[phrase@2-4]`, and `[a|b]` as a blend of two prompts.** `[a|b]` now means two actions at once (above). `(word:1.5)` weights work as before.
- **Reference sheet** (Assets panel, "Reference sheet..."): pick rows and columns, a picture and a label for each slot, and a title. You get one white sheet with the pictures in a grid, each labelled below and the title in the top left corner, saved to the media bin as a picture. Pictures come from the bin or an upload; each is cropped from the middle to the slot's shape.

## [5.2.0] "Classy Chopper" - 2026-11-08

### New: finishing touches for the finished video

- **Watermark.** Put a text or a picture from your media bin in any corner of the finished video (Settings > Engine > Post). You choose how see-through it is, how big, how far from the edge, and can add an outline. Transparent PNGs keep their transparency. Off by default.
- **Picture clean-up options** (Settings > Engine > Post, all off by default): smooth jagged edges, auto contrast, auto brightness and colour strength. They are measured once for the whole clip, so nothing flickers.
- **Interpolate** adds in-between frames to a finished video with NVIDIA's frame generation (2 to 4 times the frames, same length and sound). Needs an RTX card. Choose Never, a button next to the rating, or automatic after every render (Settings > Editor).
- **Save Video quality is now a setting, and the default is "high".** Files are bigger and take a little longer to save; colours are also very slightly brighter than before.

### New: faster and sharper generating

- **Two stages** (Settings > Engine > Sampling, MiniMax H3): the first steps run on a smaller picture and the rest finish at full size. You can pick how the picture is enlarged in between, including a learned upscaler file.
- **Veda sparse attention** (experimental, NVIDIA GPU): a quicker way to compute attention. The model file can be downloaded from Settings > Models.
- **MXFP8 for the H3 loader** (Blackwell GPUs): roughly twice the matrix speed.
- **Decode-only VAE** in the VAE Loader: a second, faster VAE used only for the last decoding step (for example the Sol-H3 VAE).
- **Resolution rule** in Project settings: choose whether Width and Height mean the short side, the long side or the pixel count. New projects start on the short side; older projects keep their size.
- **Run meter** (off by default) shows how grainy the picture is and how much it still changes each step, and compares it with your last run of the same seed. Handy to see what one setting did.

### New: LoRAs, prompts and shortcuts

- **LoRA Loader holds up to 8 LoRAs** and shows one slot at a time. Each can be applied to the first stage, the second stage or both (when Two stages is on).
- **More LoRA files work**, including the Diffusers-style files such as NVIDIA's Open-H3-Regenerate-2K refiner.
- **Prompt enhancer: Dynamic style** asks for a clear action, a change over time and secondary motion in every shot.
- **Shortcuts: Find and replace, and Common phrases.** Search every shortcut at once, preview the changes, then apply them. Common phrases lists wording shared by several shortcuts so you can fix it in one place.
- **Attention explorer: "Change the sound too"**. Turn it off to leave the sound's attention alone.

### Changed

- **Simpler scene and project settings.** These are gone because the timeline already does the job: "Pause after this clip" (drag the clip instead; edges snap to neighbours and the playhead, Shift holds it free), "Source trim" (trim with the clip's edges), the scene's "Source" choice and "Start shots" (a scene now starts from what it has: a prompt, a picture or references).
- **Video input is set once per project.** The separate per-scene copy is gone.
- **Continuity moved to Project settings** as "Pass the last frame of previous scene as reference into next scene". A shot after a clip that ends dark always starts fresh.
- **Auto Montage is gone.** Projects that already contain its pieces open normally; the pieces stay as ordinary clips.
- **"Video detail" is gone** from Post settings. Projects that used it open normally and the Log mentions it once.
- **Rarely used settings** (Video input, final render size, split preview) sit under "More".
- **Scenes that used the old Source choices** now carry on from the previous clip like the default.

### Fixed

- **Upscaling no longer makes the next generation reload all models.** Upscaling now runs beside ComfyUI instead of as one of its queued jobs. Generate is refused while an upscale model is working, so the two cannot fight over memory.
- **The Model Loader no longer reloads the model on every run** when two loaders use the same file, and a switch of attention or SLA settings frees the earlier copy cleanly.
- **No more false "memory leak" warning** from ComfyUI after a model's links were lost.
- **No more "Should have ta>=t0" warning** with SDE-type samplers. Pictures were never affected.
- **Reference numbers on media tiles match the model's inputs**, counted per kind (the first sound is R1 even if a picture holds the first slot). Pinned references go to the input of their own number, and the Log says when something had nowhere to go.
- **LoRA training says why** when it cannot copy the model, and learns from LoRA files that name their parts `lora_down` and `lora_up`.
- **The timeline is never shorter than its lanes**, the header fits one row from 1024px wide, and the guided tour follows buttons that get renamed.
- **"Build" keeps weaker drafts in play** and no longer treats the word "constructor" as a weight.
- **Stop works at once** while an Upscale or Interpolate waits for its turn.
- **Settings of removed features** are cleared when a project opens, so the Log stops warning about them.

<details>
<summary>Technical notes</summary>

- Veda needs CUDA, Triton and a predictor file in `ComfyUI/models/veda`; with Veda the SLA sparsity, rising share and references settings are not read yet.
- Two stages hands the second stage the model's clean guess; the first stage keeps the full picture's shape within a few percent.
- Save Video "high" is constant quality 16 (it was 23) with a slower encoder preset, and the finished picture is rounded to 8 bits instead of truncated. Older saved pipelines get "high".
- MXFP8 needs torch 2.10 or newer with CUDA 13; elsewhere it is skipped with a note.
- Interpolate needs nvidia-vfx 0.2 or newer; switching the add-on on updates an older copy.

</details>

## [5.1.0] "Classy Chopper" - 2026-10-27

### New

- **LTX 2.3 is tested on a real model**, with two presets: Video to Video (direct) and Image to Video (from References). Both fall back to text to video when no picture or video is given, and the run says so.
- **Attention explorer** (needs a Taste key): each run tries a small attention change per group of blocks, and your ratings decide which changes stay. Works on MiniMax H3 and LTX.
- **Noise merging** can run a whole second video beside yours from a second seed and blend the two, or mix the starting noise of two seeds.
- **Identity bank** (LTX, needs a Taste key) keeps the look of clips you like and leans on it in the next ones. Not yet tried on a real LTX run.
- **Guide frames** (LTX) use a picture from the clip before as a gentle guide for the next shot.
- **Rising sparsity** for SLA: skip fewer attention blocks early and more late. One real H3 render kept its quality.
- **Video input** (Project and Scene tabs): trim, resize, crop and re-time reference videos before the model sees them. Off by default.
- **Temporal style: auto** picks a style for each shot from its own words.
- **Flip width and height** button in Project settings.
- **Block settings know the model**: they show how many blocks it has and keep your list inside that range.

### Changed

- **LTX: the start picture is now reference picture 1** in the scene's References selector instead of a picture dropped on the timeline. Image to Video and Video to Video are separate presets.
- **On LTX, only Noise merging, the Attention explorer and Taste are offered for now.** The other enhancements come back once they have been proven on a real LTX run. Your saved settings are kept.
- **Shot camera and SLA are shown for MiniMax H3 only.**
- **Long clips use less memory** when loaded.
- **The ComfyUI log survives a restart**, and the Log window has a "Previous run" button. It also notes RAM and GPU memory use, so a run that gets killed for memory leaves a trace.

### Fixed

- Clip names in Japanese (or any non-ASCII name) no longer fail a run on a console that is not UTF-8.
- Restoring a saved pipeline no longer adds another preset's input slots.
- Older LTX Image to Video pipelines fall back to text to video when no picture is given, instead of refusing to start.
- Cancelling during a Noise merging second pass now cancels.
- Attention explorer ratings made on H3 no longer teach LTX.
- Guide frames no longer crash on an invalid position.

## [5.0.3] "Classy Chopper" - 2026-10-25

### Changed

- **FunPack is now FunUI**, with a new logo. The 5.x codename is "Classy Chopper".
- **The app moved to `/fun/`.** The old address redirects.
- Picker wheel settings are saved with the project.
- The repository moved to `digital-garbage/FunUI`; the old address redirects, so existing copies keep updating.
- Projects, imported workflows and saved settings with old `FunPack` names are renamed as they load. Nodes already placed in ComfyUI's own graph under the old names need to be added again.
- Decisiveness is now a "Commitment" slider from -10 to +10 (0 is off); a value saved by an older version resets to 0.

### New (experimental)

- **PTD few-step** (MiniMax H3 only): renders in 4 to 8 steps using a PTD adapter file from `ComfyUI/models/ptd`. Ported from ComfyUI-PTD-Compact by Nynxz.

## [5.0.2] - 2026-10-22

### Changed

- **The interface follows Apple's Human Interface Guidelines more closely**: roomier windows, better text contrast, nothing under 11px, a clearer focus ring and selection marks, and sliders whose value you can type. The Composer and Training windows open with a Genie motion.
- New README screenshots and a clearer note on how the project is maintained.

### Fixed

- "Restarting ComfyUI..." no longer appears twice.

## [5.0.1] - 2026-10-21

Fixes from real H3 runs on rented GPUs, plus the first usable LoRA training.

### New

- **LoRA training on H3** with a step-by-step window (Data, Tags, Train, Results): previews at chosen steps, Pause and Resume, Restore and continue, live graphs and downloads.
- **Gallery** for saved videos and pictures, with preview, Select mode and confirmed delete.
- **15 reference slots for H3** (9 pictures, 3 videos, 3 sounds).
- **Log colours**: a healthy log is plain text; warnings are amber and errors red.
- A bypass switch per node in Models & Pipeline, and two new Quick wheel entries: Disable all enhancements and Fix the seed.
- The loading window names the module being loaded.
- Experimental: **Distilled Flow samplers** and **Sharpness**.

### Fixed

- **Save Video no longer loses a video** when the GPU encoder will not open; it saves on the CPU and says why.
- **Reference videos and sound** now reach H3's video and audio inputs, are no longer cut to 15 seconds, and a video's own sound comes along.
- The Render button was dark with near-black text.
- RTX Super Resolution says why it failed to load.
- Taste direction works on the CPU.
- Text-only H3 runs no longer need a picture input.
- The Log survives a crash.
- Media upload shows progress and no longer stalls the server on big clips.

## [5.0.0] "Cunning Cranberry" - 2026-10-16

A rewrite from scratch, not an update of 4.x. Same idea (a video studio inside ComfyUI that learns from your ratings), new design: every feature is a module that switches on or off by itself, and a feature that fails to load is left out while the run carries on. 4.x projects are not imported.

### What it is

- **A browser studio** with a timeline, Composer (story in one box, templates, variables, shortcuts), properties, media bin, preview, takes, rating, Export and Render, project wizard and guided tour.
- **Models & Pipeline**: the pipeline as cards. Pick files, swap or add nodes, import a ComfyUI workflow. It refuses what would not build and says why.
- **Modules** (Settings > Modules): switch each feature on or off.
- **Learning from ratings**: a Taste key remembers your ratings, and the features that use them start acting once there are enough.

### Models

- **MiniMax H3**: the model v5 is built and tested around.
- **LTX 2.x**: tested on a real model, including image to video from References.
- **Stable Diffusion 1.5**: experimental.

### Features

- **Prompt enhancer**: a language model rewrites each finished prompt (has its own encoder pick).
- **Auto Montage**: builds a new cut from your rendered clips; originals stay untouched.
- **Upscale** a finished render with an upscale model or NVIDIA RTX Super Resolution.
- **LoRA training** (add-on, experimental): Data, Tags, Train and Results steps, with previews made by your own pipeline.
- **Add-ons** (spaCy, nvidia-vfx, onnxruntime) install on the next start after you switch them on.
- Many sampling and guidance features, some proven and some experimental. See Settings > Modules.

### Not carried over

- 4.x projects and settings are not imported; start new projects in 5. ComfyUI workflow files can still be imported.
- Features that never proved out stay out.

### Known limits

- `.gguf` text encoders need the ComfyUI-GGUF node pack and only work for architectures it understands.
- On H3 the scene picture sets the video's shape only; it is not a start frame.
- Some modules switch the model compiler off for the run, which costs sampling time.

See [How to start generating](docs/how-to-start.md).
