// A render-style job: start it, poll it until it stops. -> the final status ({state, detail, media}).
import { call } from "../../shell/api.js";

/** The render module's routes live under its own id. */
export const RENDER = "/api/m/render";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export async function runJob(path, body) {
  const { job_id: job } = await call("POST", path, body);
  let state;
  let blips = 0;
  // A poll the server did not answer (a restart, a proxy's page) is asked again: the job runs on without the page.
  do {
    await wait(1500);
    try { state = await call("GET", `${path}/${job}`); blips = 0; } catch (err) { if (err.status && err.status < 500 || ++blips > 20) throw err; state = { state: "running" }; }
  } while (state.state === "queued" || state.state === "running");
  return state;
}
