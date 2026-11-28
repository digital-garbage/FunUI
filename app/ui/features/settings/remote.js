// Remote Instance: drive ComfyUI and FunUI on another machine over SSH. While connected, the modules, models and queue shown
// are that machine's; projects, shortcuts and the rest of your work stay on this computer.
import { composer as c } from "../../composer/composer.js";
import { parseAddress, addressText } from "./remote_address.js";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export const remote = (app) => function mount() {
  const api = app.api;
  const body = c.region.stack({ gap: "sm", label: "Remote Instance" });
  let saved = {}, timer = null, form = null, misses = 0;

  const field = (label, props, hint) => {
    const input = c.input.md({ label, ...props });
    return { input, row: c.settingsRow.default({ label, hint, control: input }) };
  };

  function draw(s) {
    if (s.state === "on") {
      body.set([
        c.hint.default({ text: `Connected to ${s.target}. This page now controls ComfyUI and FunUI on that machine: the modules, models and queue you see are theirs. Projects, shortcuts and your other work stay on this computer.` }),
        c.button.sm({ label: "Disconnect", tone: "primary", onClick: async () => { await api.disconnectRemote(); location.reload(); } }),
      ]);
      return;
    }
    if (s.state === "working") {
      body.set([c.label.section({ text: "Connecting" }), ...s.steps.map((t, i) => c.hint.default({ text: `${i === s.steps.length - 1 ? "▸" : "✓"} ${t}` })),
        c.hint.default({ text: "Installing or updating FunUI on the other machine can take a few minutes." }),
        c.button.sm({ label: "Cancel", tone: "ghost", onClick: async () => { try { await api.disconnectRemote(); } catch { /* the next look at the state says what is true */ } follow(); } })]);
      return;
    }
    const L = s.login || {};
    saved = L;
    const address = field("Address", { value: addressText(L), placeholder: "host:port or the ssh line" }, "Where the machine is. The whole ssh line from a rental site works.");
    const user = field("Login", { value: L.user || "root", placeholder: "root" });
    const key = field("Key file", { value: L.key || "", placeholder: "~/.ssh/id_ed25519" }, "Leave empty to use the password, or your default keys.");
    const pass = field("Password", { value: "", type: "password" }, "Or the passphrase of the key. It is not saved.");
    form = { address, user, key, pass };
    const go = c.button.sm({ label: "Connect", tone: "primary", onClick: connect });
    body.set([
      c.hint.default({ text: "Connect over SSH and this page controls ComfyUI and FunUI on another machine. FunUI there is installed or updated to this version first." }),
      address.row, user.row, key.row, pass.row,
      ...(s.state === "failed" && s.message ? [c.hint.default({ text: s.message })] : []),
      ...(s.paramiko === false ? [c.hint.default({ text: "The SSH library is not installed here. Update FunUI's requirements, then try again." })] : []),
      go,
    ]);
  }

  async function connect() {
    const parsed = parseAddress(form.address.input.value);
    if (!parsed.host) return c.toast.warn({ text: "Fill in the address first." });
    try {
      await api.connectRemote({
        host: parsed.host, port: parsed.port || 22, user: parsed.user || form.user.input.value || "root",
        key: parsed.key || form.key.input.value, password: form.pass.input.value,
      });
    } catch (err) { return c.toast.warn({ text: `Could not start the connection: ${err.message}` }); }
    follow();
  }

  async function follow() {
    clearTimeout(timer);
    let s;
    try { s = await api.remote(); misses = 0; }
    catch (err) {                                                    // this computer's FunUI did not answer: say so, never sit on "Connecting"
      if (++misses < 4) { timer = setTimeout(follow, 1500); return; }
      return draw({ state: "failed", message: `This computer's FunUI is not answering (${err.message}).`, login: saved, steps: [] });
    }
    if (s.state === "on") return location.reload();                 // everything on the page is now the remote machine's
    draw(s);
    if (s.state === "working") timer = setTimeout(follow, 1000);
  }

  follow();
  return body;
};
