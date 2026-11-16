import { mount } from "./view.js";

export default {
  id: "temp",
  mount: "settings",
  needs: ["dom"],
  setup: ({ host, app }) => host.add({
    id: "temp", group: "System", title: "Generated & Output Files", subtitle: "What was generated since ComfyUI started (its temp folder, emptied on every start), and what is in ComfyUI's output folder.",
    keywords: "gallery generated saved temp files output directory videos pictures delete", icon: "▥", mount: (ctx) => mount({ ...ctx, dom: app.dom, onDeleted: () => app.say && app.say("files.deleted") }),
  }),
};
