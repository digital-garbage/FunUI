// Temporal style "auto": each shot gets the style its own words ask for. The choice is made here, per shot, and sent as the concrete
// style in that run's modifier settings; the module itself only knows concrete styles.
import { styleFor } from "./intent.js";

export default {
  id: "temporal",
  mount: "menubar.menus",
  needs: ["pipeline", "inputHooks"],
  modules: ["temporal_style"],
  setup({ app }) {
    const hook = async ({ scene, slots }) => {
      if (app.pipeline.allOff?.(slots)) return {};
      const values = app.pipeline.currentValues(slots) || {};
      if ((values.temporal_style || {}).style !== "auto") return {};
      const slot = (slots || []).find((s) => s.node === "FunUIModifierSettings");
      if (!slot) return {};
      const style = styleFor(scene.text);
      const settings = JSON.stringify({ ...values, temporal_style: { ...values.temporal_style, style } });
      return { inputs: { [slot.id]: { settings } }, notes: [`Temporal style auto: ${style} for this shot.`] };
    };
    app.inputHooks.push(hook);
    return () => { const i = app.inputHooks.indexOf(hook); if (i >= 0) app.inputHooks.splice(i, 1); };
  },
};
