// The Project tab: what the whole project makes.
import { composer as c } from "../../composer/composer.js";
import { DRIVEN, isVideoClip, segments, snapFrames } from "../../shell/scenes.js";
import { gridOf, offersLength } from "../../shell/frame_grid.js";
import { card } from "./scene.js";
import { projectVideoInput } from "./videoinput.js";
import { fold } from "./fold.js";
import { settingRow, whenSatisfied } from "../settings/field.js";

const BASES = [{ value: "short", label: "Short side" }, { value: "long", label: "Long side" }, { value: "area", label: "Pixel count" }];
const BASIS_HINT = { short: "With Short side: the clip's shorter side is the smaller of Width and Height; the scene's picture sets the shape. 1024 × 1024 and a 16:9 picture make 1824 × 1024.",
  long: "With Long side: the clip's longer side is the larger of Width and Height; the scene's picture sets the shape. 1024 × 1024 and a 16:9 picture make 1024 × 576.",
  area: "With Pixel count only: the clip has about Width × Height pixels; the scene's picture sets the shape. 1024 × 1024 and a 16:9 picture make 1376 × 768." };

/** For a model that fits the scene picture's shape to Width and Height: what those two numbers mean. */
const basis = (p, slots) => (slots || []).some((s) => (s.roles || []).some((r) => r.at === "project.resolution_basis")) ? c.field.default({ label: "Resolution is the", hint: BASIS_HINT[p.project.resolution_basis] || BASIS_HINT.short,
  control: c.select.md({ label: "Resolution is the", options: BASES, value: BASES.some((b) => b.value === p.project.resolution_basis) ? p.project.resolution_basis : "area", onChange: (v) => p.setField("resolution_basis", v) }) }) : null;

// What the pipeline lets the project decide (width, length, rate...): one whole-number control per slot input
// that declares the role. A role that `drives` length or rate edits the number the timeline draws.
const videoControls = (p, slots) => (slots || []).flatMap((slot) => (slot.roles || []).filter((r) => r.at === "project.video" && r.input).map((role) => {
  const field = DRIVEN[role.drives];
  const now = field ? p.project[field] : p.video[role.input] ?? p.video[role.was] ?? (slot.inputs || {})[role.input];
  if (!Number.isInteger(now)) return null;                  // the project file keeps whole numbers only
  const label = role.label || role.input, grid = role.drives === "frames" ? gridOf(slots) : null;
  // On a model with a frame grid the arrows walk it (H3: 22, 39, 56 …) and a typed number lands on it: what runs is what shows.
  return { key: field || role.input, label, now, field: c.field.default({ label, hint: grid ? `This model makes ${grid.step}k+${grid.base} frames; other numbers snap to the nearest.` : undefined,
    control: c.number.md({ label, min: grid ? grid.step + grid.base : 1, max: role.drives === "fps" ? 1000 : 16384, step: grid ? grid.step : 1, precision: 0, value: now,
      snap: grid ? (v) => snapFrames(v, "round", grid) : undefined, onChange: (v) => (field ? p.setField(field, v) : p.setVideo(role.input, v)) }) }) };
}).filter(Boolean)).filter((r, i, all) => all.findIndex((x) => x.key === r.key) === i);

const num = (p, label, key, fallback) => ({ key, label, field: c.field.default({ label, control: c.number.md({ label, min: 1, max: key === "frame_rate" ? 1000 : 16384, step: 1, precision: 0,
  value: p.project[key] || fallback, onChange: (v) => p.setField(key, v) }) }) });

/** What each shot's prompt becomes at generation (anchor + shortcuts + $variables + postfix), asked for on demand. */
function splitPreview(p, api) {
  const out = c.region.stack({ gap: "xs" });
  const show = async () => {
    out.set([c.hint.default({ text: "Working…" })]);
    try {
      const open = p.project, roots = open.scenes.filter((s) => !s.excluded && !(s.cut_offset_frames > 0) && !(s.source && s.source.type === "video"));
      const lines = await Promise.all(roots.map(async (s, i) => ({ n: open.scenes.indexOf(s) + 1, text: (await api.expandPrompt({ text: s.text || "", anchor: open.anchor, postfix: open.postfix, postfix_enabled: open.postfix_enabled, variables: open.variables })).text })));
      out.set(lines.length ? lines.map((l) => c.field.default({ label: `S${l.n}`, control: c.text.sm({ text: l.text || "(empty)" }) })) : [c.hint.default({ text: "No scenes to preview." })]);
    } catch (err) { out.set([c.hint.default({ text: `Could not build the preview: ${err.message}` })]); }
  };
  return c.region.stack({ gap: "xs", children: [c.button.sm({ label: "Refresh preview", tone: "ghost", onClick: show }), out] });
}

/** The final render's size can come from the first clip, the project, or any rendered clip's own. */
function sizes(open) {
  const rendered = segments(open).filter((s) => s.kind === "scene" && !s.scene.excluded && (open.scene_renders || {})[s.scene.id]);
  const list = [{ value: "", label: "The first clip's own size" }, { value: "project", label: `The project's size (${open.width || 768}×${open.height || 512})` },
    ...rendered.map((s, i) => ({ value: s.scene.id, label: `Clip ${i + 1}'s own size` }))];
  if (open.export_size_from && !list.some((o) => o.value === open.export_size_from)) list.push({ value: open.export_size_from, label: "A clip that is no longer here (the first clip is used)" });
  return list;
}

/** Shots set to Custom length ignore the project's Frames: say so, and offer to put them back. */
function custom(p, open) {
  const gen = open.scenes.filter((s) => !isVideoClip(s)), over = gen.filter((s) => s.frames_mode === "custom");
  if (!over.length) return [];
  return [c.banner.warn({ text: over.length >= gen.length ? `Frames doesn't reach any shot — all ${over.length} are set to Custom.` : `Frames doesn't reach ${over.length} of ${gen.length} shots — they're set to Custom.`,
    action: { label: "Use project length everywhere", onClick: () => p.edit((pr) => { let hit = false; for (const s of pr.scenes) if (!isVideoClip(s) && s.frames_mode === "custom") { s.frames_mode = "project"; hit = true; } return hit; }) } })];
}

/** Swaps width and height in one undo step: portrait from landscape and back. Only when the pipeline gives both a number. */
function flip(p, size) {
  const w = size.find((x) => /width/i.test(x.key)), h = size.find((x) => /height/i.test(x.key));
  if (!w || !h) return null;
  return c.button.sm({ label: "⇄ Flip width and height", tone: "ghost", title: `Make it ${h.now}×${w.now}`, onClick: () => p.edit((pr) => {
    pr.video = pr.video || {};
    pr.video[w.key] = h.now;
    pr.video[h.key] = w.now;
    return h.now !== w.now;
  }) });
}

/** Module settings that belong in Project settings (they declare `home: "project"`), for the modules acting on this pipeline: [[module, name, spec]]. */
const homedSettings = (ps) => !ps || !ps.activeModules ? [] : ps.activeModules().filter((m) => ps.useful(m)).flatMap((m) => Object.entries(m.settings || {}).filter(([, spec]) => spec.home === "project").map(([name, spec]) => [m, name, spec]));
/** The values behind those rows, so the panel redraws when one changes. */
export const homed = (app) => { const ps = app.pipeline, v = ps && ps.currentValues ? ps.currentValues() : {}; return homedSettings(ps).map(([m, name]) => (v[m.id] || {})[name]); };

/** Continuity's choices, in a card of their own (only while that module acts on the pipeline). */
function frameCarry(app) {
  const ps = app.pipeline;
  if (!ps || !ps.activeModules) return [];
  const values = ps.currentValues();
  const rows = homedSettings(ps).filter(([m, , spec]) => whenSatisfied(values, m.id, spec.when)).map(([m, name, spec]) => settingRow(m.id, name, spec, values, (id, n, v) => ps.setModuleValue(id, n, v)));
  return rows.length ? [c.label.section({ text: "Scenes" }), card(...rows)] : [];
}

export function projectRows(p, slots, api, app = {}) {
  const open = p.project;
  if (!open) return [c.emptyState.default({ icon: "▭", title: "No project", hint: "Open or create one." })];
  const controls = videoControls(p, slots);
  const pick = (...keys) => keys.map((k) => controls.find((x) => x.key === k)).filter(Boolean);
  const frames = pick("num_frames_per_scene")[0] || num(p, "Frames / scene", "num_frames_per_scene", 97);
  const fps = pick("frame_rate")[0] || num(p, "FPS", "frame_rate", 25);
  const size = controls.filter((x) => x !== frames && x !== fps);
  const per = (open.num_frames_per_scene || 97) / (open.frame_rate || 25);
  const timed = offersLength(slots);               // a pipeline that holds a picture for a fixed time takes no length or rate
  return [
    c.label.section({ text: "Project" }),
    card(c.field.default({ label: "Project name", control: c.input.md({ label: "Project name", value: open.name || "", onCommit: (v) => p.rename(v) }) })),
    c.label.section({ text: "Video settings" }),
    card(timed ? c.field.row({ fields: [frames.field, fps.field] }) : null,
      timed ? c.hint.default({ text: `≈ ${per.toFixed(2)} s per shot` }) : null,
      size.length ? c.field.row({ fields: size.map((x) => x.field) }) : null,
      flip(p, size),
      basis(p, slots),
      ...custom(p, open)),
    ...frameCarry(app),
    c.label.section({ text: "Prompt" }),
    card(c.field.default({ label: "Anchor", hint: "Put in front of every scene's prompt.", control: c.textarea.md({ label: "Anchor", value: p.anchor, rows: 2, onInput: (v) => p.setAnchor(v) }) }),
      c.field.default({ label: "Negative prompt", hint: "What to steer away from. A model that ignores negative prompts (MiniMax H3) uses it only with Shadow negative or Negative erase on.", control: c.textarea.md({ label: "Negative prompt", value: p.negative, rows: 2, onInput: (v) => p.setNegative(v) }) }),
      c.toggle.default({ label: "Postfix", hint: "Appended to every scene (a style or quality tag, say).", checked: open.postfix_enabled !== false, onChange: (v) => p.setField("postfix_enabled", v) }),
      c.textarea.md({ label: "Postfix", value: p.postfix, rows: 2, disabled: open.postfix_enabled === false, onInput: (v) => p.setPostfix(v) })),
    c.label.section({ text: "More" }),
    fold("project-export", "Final render size", [card(c.field.default({ label: "Final render size", hint: "The size of the exported video: the first clip's, the project's, or any rendered clip's own.", control: c.select.md({ label: "Final render size", options: sizes(open), value: open.export_size_from || "",
      onChange: (v) => p.setField("export_size_from", v) }) }))]),
    ...projectVideoInput(p),
    fold("project-split", "Split preview (generation prompt)", [splitPreview(p, api)]),
  ];
}
