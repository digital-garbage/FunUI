// How a picked reference video is got ready for the model (the loader choices: size, crop, rate, skip, cut, every Nth, cap).
// The project keeps one set and every scene its own; `video_input_from` says whose is used.
import { effFps } from "./scenes.js";

export const FROM = [{ value: "auto", label: "Automatic (the project's when on, else the scene's)" }, { value: "project", label: "The project's" }, { value: "scene", label: "Each scene's own" }];
export const FITS = [{ value: "crop", label: "Fill (crop the overflow)" }, { value: "pad", label: "Fit (black bars)" }, { value: "stretch", label: "Stretch" }];
export const RATES = [{ value: "project", label: "The project's FPS" }, { value: "source", label: "The video's own" }, { value: "custom", label: "A number" }];

export const BLANK = { on: false, fit: "crop", fps_from: "project", fps: 24, skip_first: 0, cut_last: 0, every_nth: 1, frame_cap: 0, width: 0, height: 0, multiple_of: 32, crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0 };

/** The settings that apply to this scene, or null when none is switched on (the video then loads as it always did). */
export function activeVideoInput(project, scene) {
  const own = { ...BLANK, ...((project || {}).video_input || {}) }, mine = { ...BLANK, ...((scene || {}).video_input || {}) };
  const from = (project || {}).video_input_from || "auto";
  const use = from === "project" ? own : from === "scene" ? mine : own.on ? own : mine;
  return use.on ? use : null;
}

/** The JSON the loader node takes. */
export function videoInputJson(use, project, scene) {
  const fps = use.fps_from === "source" ? 0 : use.fps_from === "custom" ? use.fps : effFps(scene, project);
  const { on, fps_from, ...rest } = use;
  return JSON.stringify({ ...rest, fps });
}
