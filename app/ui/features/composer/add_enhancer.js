// "Add enhancer": a pipeline without a prompt enhancer gets one, wired the way the shipped presets have it: the enhancer reads the
// prompt (and the text encoder), and the encoder that used to read the prompt reads the enhancer's answer instead.
const NODE = "FunUIEnhancePrompt";
const isLink = (v) => Array.isArray(v) && v.length === 2 && typeof v[0] === "string";

/** -> { slots } the pipeline with the enhancer in, or { why } when there is no single prompt encoder to put it before. */
export function withEnhancer(slots) {
  const all = slots || [];
  if (all.some((s) => s.node === NODE)) return { why: "There is a prompt enhancer already." };
  const target = all.filter((s) => (s.roles || []).some((r) => r.at === "generation.prompt" && r.input === "text"));
  if (target.length !== 1) return { why: target.length ? "More than one slot takes the prompt, so there is no single place for the enhancer." : "This pipeline has no prompt slot to put an enhancer in front of." };
  const t = target[0], clip = (t.inputs || {}).clip;
  if (!isLink(clip)) return { why: `The prompt slot (${t.id}) has no text encoder wired, so the enhancer could not be given one.` };
  const hub = all.find((s) => s.node === "FunUIModifierSettings");          // where prompt steps such as Shot camera are switched on
  let id = "enhance";
  for (let n = 2; all.some((s) => s.id === id); n++) id = `enhance_${n}`;
  const role = t.roles.find((r) => r.at === "generation.prompt" && r.input === "text");
  const enhancer = {
    id, group: t.group || "Preparation", node: NODE,
    roles: [{ at: "generation.prompt", input: "text", label: role.label || "Prompt" }, { at: "generation.prompt", input: "enabled", label: "Enhance prompt" }],
    inputs: { clip: [...clip], text: (t.inputs || {}).text || "", enabled: false, ...(hub ? { settings: [hub.id, 0] } : {}) },
  };
  const out = [];
  for (const s of all) {
    if (s === t) {
      out.push(enhancer);
      out.push({ ...s, roles: s.roles.filter((r) => r !== role), inputs: { ...s.inputs, text: [id, 0] } });
    } else out.push(s);
  }
  return { slots: out };
}
