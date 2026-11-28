import test from "node:test";
import assert from "node:assert/strict";
import { parseAddress, addressText } from "./remote_address.js";

test("plain host, host:port and user@host:port", () => {
  assert.deepEqual(parseAddress("gpu.example.com"), { host: "gpu.example.com" });
  assert.deepEqual(parseAddress("gpu.example.com:2222"), { host: "gpu.example.com", port: 2222 });
  assert.deepEqual(parseAddress("me@10.0.0.5:22"), { user: "me", host: "10.0.0.5", port: 22 });
});

test("the line a rental site gives is read whole, and the tunnel part is ignored", () => {
  assert.deepEqual(parseAddress("ssh -p 31189 root@ssh4.vast.ai -L 8080:localhost:8080"), { port: 31189, user: "root", host: "ssh4.vast.ai" });
  assert.deepEqual(parseAddress("ssh -i ~/.ssh/k -p 2200 root@h.io"), { key: "~/.ssh/k", port: 2200, user: "root", host: "h.io" });
});

test("nonsense ports are dropped, empty is empty", () => {
  assert.deepEqual(parseAddress("h:99999"), { host: "h" });
  assert.deepEqual(parseAddress("  "), {});
  assert.equal(addressText({ host: "h", port: 22 }), "h");
  assert.equal(addressText({ host: "h", port: 5 }), "h:5");
});
