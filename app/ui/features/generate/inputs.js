// What one run sends: the scene's prompt (expanded with anchor/postfix/$variables), the project's video
// settings, the scene's picked media -- each placed by the ROLE a pipeline slot declares, never by slot id.
import { sceneReferences, wireReferences, routeFor } from "../../shell/reference_wiring.js";
import { kinds, sounds } from "../../shell/bin.js";
import { DRIVEN } from "../../shell/scenes.js";
import { activeVideoInput, videoInputJson } from "../../shell/video_input.js";

export const rolesAt = (slots, at) => (slots || []).flatMap((s) => (s.roles || []).filter((r) => r.at === at).map((role) => ({ slot: s, role })));

/** `frames`: how long this unit is (its clips together); a fresh `seed` per call so a re-roll is a different result (`seed: null` keeps the node's own). */
export async function buildInputs({ project, scene, slots, expand, nodeInfo, frames, hooks = [], prefix = [], seed = () => Math.floor(Math.random() * 2 ** 31) }) {
  const raw = {};
  const put = (slot, name, value) => { raw[slot.id] = { ...raw[slot.id], [name]: value }; };

  for (const { slot, role } of (slots || []).flatMap((s) => (s.roles || []).filter((r) => (r.at || "").startsWith("project.") && r.input).map((role) => ({ slot: s, role })))) {
    const v = role.at === "project.negative" ? project.negative : DRIVEN[role.drives] ? (role.drives === "frames" && frames ? frames : project[DRIVEN[role.drives]]) : (project.video || {})[role.input];
    if (v !== undefined && v !== null && v !== "") put(slot, role.input, v);
  }
  if (seed) for (const { slot, role } of rolesAt(slots, "generation.seed")) put(slot, role.input, seed());      // null: the seed node's own number runs
  for (const slot of (slots || []).filter((s) => s.node === "FunUIShotCamera")) put(slot, "project", project.id || "");      // what the node keeps per project (the words focus infusion draws on), also for pipelines saved before it was asked for
  const source = rolesAt(slots, "assets.source_image")[0];
  const plan = sceneReferences(scene, project.fixed_references);
  // A pipeline that takes whole media where it takes a picture (LTX): with no start picked, the first reference is the start:
  // a VIDEO first (it is then not also sent as a reference: nothing takes it there), else the first picture (which stays a
  // reference too, so an identity input still gets it). A video-only pipeline (video-to-video) looks at videos alone and is
  // never blanked by "From a prompt".
  const takesVideo = Boolean(source && source.role.video_frames), videoOnly = Boolean(source && source.role.video_only);
  const wanted = (id) => kinds.get(id) === "video" || (!videoOnly && kinds.get(id) === "image");
  const blank = project.generation_mode === "t2v" && !videoOnly;
  const fromRefs = takesVideo && !scene.source_image && !blank
    ? plan.ids.find((id) => kinds.get(id) === "video") || (videoOnly ? null : plan.ids.find((id) => wanted(id))) : null;
  if (fromRefs && kinds.get(fromRefs) === "video") plan.ids = plan.ids.filter((id) => id !== fromRefs);
  const startId = takesVideo ? (blank ? "" : scene.source_image || fromRefs || "") : project.generation_mode === "t2v" ? "" : scene.source_image;
  if (source) put(source.slot, "media_id", startId || "");    // "from a prompt" starts every shot without a picture
  // A pipeline that can take a whole video where it takes a picture (LTX) says so on the role: a picked VIDEO then
  // supplies its frames (prepared as Video input says); the preset wires the frames output where the picture goes (a
  // wired input cannot be overridden per run), and that output is the first frame alone until `full` is set.
  if (takesVideo && startId && kinds.get(startId) === "video") {
    put(source.slot, "full", true);
    const use = activeVideoInput(project, scene);
    if (use) put(source.slot, "prepare", videoInputJson(use, project, scene));
  }
  // A project saved before the preset said so still has this shape: a picture-resizing slot is enough to know a stand-in fits.
  const resizer = (slots || []).find((s) => s.node === "ImageTransformKJ"), enhancer = (slots || []).find((s) => s.node === "FunUIEnhancePrompt");
  const like = source && (source.role.blank_like || (resizer && { slot: resizer.id, width: "target_width", height: "target_height", ...(enhancer ? { unsend: { slot: enhancer.id, input: "use_image" } } : {}) }));
  if (like && !raw[source.slot.id].media_id) {          // text-only: a plain picture of the size the pipeline asks for stands in
    const target = (slots || []).find((s) => s.id === like.slot) || { inputs: {} };
    const size = (name) => (raw[target.id] || {})[name] ?? (target.inputs || {})[name];
    put(source.slot, "blank_width", Number(size(like.width)) || 0);
    put(source.slot, "blank_height", Number(size(like.height)) || 0);
    const off = like.unsend && (slots || []).find((s) => s.id === like.unsend.slot);      // a stand-in is not a picture to show anyone
    if (off) put(off, like.unsend.input, false);
  }
  // A reference is placed from what the nodes offer, so a pipeline someone edited needs nothing declared for it.
  const needed = plan.ids.length > 0;
  const info = new Map();
  if (needed && nodeInfo) {
    const classes = new Set();
    for (const s of slots || []) if ((s.roles || []).some((r) => r.wireTo)) {
      classes.add(s.node);
      for (const r of s.roles) if (r.wireTo) { const t = (slots || []).find((x) => x.id === r.wireTo.slot); if (t) classes.add(t.node); }
    }
    await Promise.all([...classes].map((c) => Promise.resolve(nodeInfo(c)).then((d) => info.set(c, d), () => {})));
  }
  const routeOf = (kind, slot, role) => {
    const target = role && role.wireTo && (slots || []).find((x) => x.id === role.wireTo.slot);
    return target ? routeFor(kind, info.get(target.node), info.get(slot.node), target.id) : null;
  };
  const { overrides, unwired, left } = wireReferences(plan.ids, slots, source ? [source.slot.id] : [], plan.pins, (id) => kinds.get(id), routeOf, (id) => sounds.has(id));
  for (const [id, fields] of Object.entries(overrides)) raw[id] = { ...raw[id], ...fields };
  // A reference video is got ready as the project (or the scene) says; with nothing switched on it loads as before.
  const prepare = activeVideoInput(project, scene);
  if (prepare) for (const [id, fields] of Object.entries(overrides)) if (fields.full && fields.media_id && kinds.get(fields.media_id) === "video") raw[id].prepare = videoInputJson(prepare, project, scene);
  // A picked start picture that no slot takes would otherwise be ignored without a word.
  const ignored = !source && scene.source_image && project.generation_mode !== "t2v";
  if (ignored) left.push({ id: scene.source_image, at: "assets.source_image", why: "this pipeline starts from no picture" });

  const prompt = rolesAt(slots, "generation.prompt")[0];
  let failedExpand = false;
  if (prompt) {
    const text = scene.text || "";
    const body = await expand({ text, anchor: [...prefix, project.anchor].filter((x) => x && String(x).trim()).join(" "), postfix: project.postfix, postfix_enabled: project.postfix_enabled, variables: project.variables })
      .catch(() => null);         // a failing prompt-craft feature must not block the run: send what was typed
    put(prompt.slot, prompt.role.input, body && typeof body.text === "string" ? body.text : text);
    if (!(body && typeof body.text === "string")) failedExpand = true;
  }
  // Features that add inputs of their own (a Chat comment for the enhancer, say) say so through a hook: { inputs: {slot: {name: value}}, notes: [text] }.
  const noVideo = takesVideo && videoOnly && !startId ? ["This pipeline works from a video: pick one as the start, or add it under References."] : [];
  const notes = failedExpand ? ["The prompt could not be expanded (shortcuts, $variables, anchor, postfix): the text was sent as typed."] : [];
  notes.push(...noVideo);
  for (const hook of hooks) {
    let out;
    try { out = (await hook({ project, scene, slots })) || {}; } catch (err) { notes.push(`An add-on's input could not be added to this run: ${err.message}`); continue; }
    for (const [id, fields] of Object.entries(out.inputs || {})) raw[id] = { ...raw[id], ...fields };
    notes.push(...(out.notes || []));
  }
  // Not refused (a picture alone may be the point), but a shot with nothing to go on is said, not silently made.
  // Judged on what is SENT (a carried frame counts; a picture t2v blanks, or one this pipeline has no input for, does not).
  const picture = (source && (raw[source.slot.id] || {}).media_id) || Object.keys(overrides).length;
  if (prompt && !String((raw[prompt.slot.id] || {})[prompt.role.input] ?? "").trim() && !picture) notes.push("This shot has no text and no start picture, so the model makes whatever it likes.");
  return { inputs: raw, unwired: unwired + (ignored ? 1 : 0), left, noPrompt: !prompt, notes };
}
