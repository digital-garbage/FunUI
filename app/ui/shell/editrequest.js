// The model proposes a change to one of its own system prompts: show the change as a before/after diff and ask.
import { composer as c } from "../composer/composer.js";
import { lineDiff } from "./diff.js";

/** The names the person knows each rule by (the Enhance tab's labels). */
export const RULE_LABELS = { instructions: "Rewrite instructions", marks_prompt: "Markup rule", fixed_prompt: "Fixed text around it", reference_intro: "Reference intro",
  conversation_prompt: "Conversation, plain encoders", turns_prompt: "Conversation, Gemma 4", revision_prompt: "Comments", rating_prefix: "Rating line opening",
  chat_prompt: "Chat answers", notes_prompt: "Notes rule", compact_prompt: "Conversation notes", edit_prompt: "Editing its own rules" };

/** Ask whether a change to a system prompt is allowed. `request`: { field, old, new, reason }. -> Promise<boolean>; closing the window means no. */
export function askEdit(request) {
  const rows = lineDiff(request.old, request.new).filter((r) => r.text !== "" || r.t !== " ");
  const lines = rows.map((r) => `${r.t === " " ? "  " : `${r.t} `}${r.text}`), tones = rows.map((r) => (r.t === "+" ? "add" : r.t === "-" ? "del" : null));
  const view = c.code.block({ text: "", label: "Changes to the system prompt" });
  view.setLines(lines, tones);
  const label = RULE_LABELS[request.field] || request.field;
  return new Promise((resolve) => {
    let settled = false;
    const done = (yes) => { if (settled) return; settled = true; win.close("answered"); resolve(yes); };
    const no = c.button.md({ label: "Disallow", tone: "ghost", onClick: () => done(false) }), yes = c.button.md({ label: "Allow", tone: "primary", onClick: () => done(true) });
    const win = c.modal.generic({ title: "Allow changes to system prompt?", subtitle: `${label}${request.reason ? `: ${request.reason}` : ""}`, size: "lg", closeOnOutside: false, closeOnEsc: false,
      body: c.region.stack({ gap: "sm", children: [c.hint.default({ text: `Prompt: ${label}. Lines with - are removed, lines with + are added.` }), view] }), actions: [no, yes],
      onClose: () => { if (!settled) { settled = true; resolve(false); } } });
  });
}
