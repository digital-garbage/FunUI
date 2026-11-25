// The question the prompt enhancer may ask before it rewrites a prompt: a small window with an input, Cancel and Send. The run waits for it;
// Cancel (or closing the window) rewrites without an answer. Its own feature, absent when the prompt enhancer module is.
import { composer as c } from "../../composer/composer.js";

export default {
  id: "briefing",
  mount: "menubar.menus",                                // no visible part: it only needs a place to be set up
  needs: ["api", "generate"],
  modules: ["conditioning_prompt_enhancer"],
  setup({ app }) {
    let shown = "", win = null;
    const off = app.generate.subscribe((st) => {
      const b = st && st.briefing;
      if (!b) { if (win) { win.close("cancelled"); win = null; } shown = ""; return; }      // the run ended or moved on: the window goes
      if (b.id === shown) return;
      shown = b.id;
      win = c.modal.prompt({ title: "A question before it writes", label: b.question, placeholder: "Your answer", confirmLabel: "Send", cancelLabel: "Cancel",
        onResult: (text) => { win = null; app.api.enhancerBriefingAnswer(b.id, text || "").catch(() => {}); } });
    });
    return () => { off(); if (win) win.close("cancelled"); };
  },
};
