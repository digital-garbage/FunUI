// The "Video input" card: how a picked reference video is got ready (size, crop, rate, skip, cut, every Nth, cap).
// One card for the project and one per scene; the project's dropdown says whose is used.
import { composer as c } from "../../composer/composer.js";
import { card } from "./scene.js";
import { BLANK, FITS, FROM, RATES, activeVideoInput } from "../../shell/video_input.js";

/** `use`: the settings shown; `save(next)` stores them. `size`: the project's width/height, offered as a one-tap target. */
function settingsCard(use, save, size) {
  const set = (key) => (v) => save({ ...use, [key]: v });
  const n = (label, key, { min = 0, max = 100000, hint, step = 1 } = {}) => c.field.default({ label, hint, control: c.number.md({ label, min, max, step, precision: 0, value: use[key], onChange: set(key) }) });
  return card(
    c.toggle.default({ label: "Prepare video inputs", hint: "Size, crop, frame rate and which frames a picked reference video gives the model. Off: the video loads whole at 24 fps.", checked: use.on, onChange: set("on") }),
    ...(use.on ? [
      c.label.section({ text: "Frames" }),
      c.field.row({ fields: [n("Skip first", "skip_first", { hint: "Frames dropped from the start." }), n("Cut last", "cut_last", { hint: "Frames dropped from the end." })] }),
      c.field.row({ fields: [n("Every Nth frame", "every_nth", { min: 1, max: 1000, hint: "1 keeps every frame." }), n("Frame cap", "frame_cap", { hint: "0 = no limit." })] }),
      c.field.row({ fields: [c.field.default({ label: "Frame rate", control: c.select.md({ label: "Frame rate", options: RATES, value: use.fps_from, onChange: set("fps_from") }) }),
        use.fps_from === "custom" ? n("FPS", "fps", { min: 1, max: 1000 }) : null].filter(Boolean) }),
      c.label.section({ text: "Picture" }),
      c.field.row({ fields: [n("Width", "width", { max: 16384, hint: "0 keeps the video's own." }), n("Height", "height", { max: 16384, hint: "0 keeps the video's own." })] }),
      c.button.sm({ label: "⇄ Flip width and height", tone: "ghost", onClick: () => save({ ...use, width: use.height, height: use.width }) }),
      size ? c.button.sm({ label: `Use the project's size (${size[0]}×${size[1]})`, tone: "ghost", onClick: () => save({ ...use, width: size[0], height: size[1] }) }) : null,
      c.field.row({ fields: [c.field.default({ label: "When the shape differs", control: c.select.md({ label: "When the shape differs", options: FITS, value: use.fit, onChange: set("fit") }) }),
        n("Round to a multiple of", "multiple_of", { min: 1, max: 256, hint: "Most models want 32." })] }),
      c.label.section({ text: "Crop the edges first (%)" }),
      c.field.row({ fields: [n("Left", "crop_left", { max: 90 }), n("Right", "crop_right", { max: 90 })] }),
      c.field.row({ fields: [n("Top", "crop_top", { max: 90 }), n("Bottom", "crop_bottom", { max: 90 })] }),
    ] : []));
}

export function projectVideoInput(p) {
  const open = p.project, use = { ...BLANK, ...(open.video_input || {}) };
  const v = open.video || {}, size = [v.width || v.target_width || open.width, v.height || v.target_height || open.height];
  return [
    c.label.section({ text: "Video input" }),
    card(c.field.default({ label: "Reference videos use", hint: "Where the preparation comes from. Automatic: the project's when it is on, otherwise each scene's own.",
      control: c.select.md({ label: "Reference videos use", options: FROM, value: open.video_input_from || "auto", onChange: (x) => p.setField("video_input_from", x) }) })),
    settingsCard(use, (next) => p.setField("video_input", next), size[0] && size[1] ? size : null),
  ];
}

export function sceneVideoInput(p, sc) {
  const open = p.project, use = { ...BLANK, ...(sc.video_input || {}) }, live = activeVideoInput(open, sc);
  const from = open.video_input_from || "auto";
  return [
    c.label.section({ text: "Video input" }),
    from === "project" || (from === "auto" && (open.video_input || {}).on)
      ? c.hint.default({ text: "The project's video input settings are used; this scene's own are ignored. Change that in Project settings." })
      : null,
    settingsCard(use, (next) => p.setScene(sc.id, "video_input", next), null),
    live ? null : c.hint.default({ text: "Switched off: reference videos load whole at 24 fps." }),
  ].filter(Boolean);
}
