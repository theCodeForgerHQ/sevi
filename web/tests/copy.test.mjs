// Issue #51: copy.js reads text only from the generated JSON; a missing key never crashes.
import assert from "node:assert/strict";
import { test } from "node:test";

globalThis.sessionStorage ??= { getItem: () => null, setItem: () => {} };
const copy = await import("../static/js/copy.js");

test("no FALLBACK table is exported", () => {
  assert.equal(copy.FALLBACK, undefined);
});

test("t fills placeholders from the loaded table", () => {
  copy._setTable({ "ui.test.round": "Round {k} of {n}" }, "en");
  assert.equal(copy.t("ui.test.round", { k: 2, n: 5 }), "Round 2 of 5");
  assert.equal(copy.t("ui.test.round"), "Round {k} of {n}");
});

test("missing key shows the key name and warns once", () => {
  copy._setTable({}, "ta");
  const warn = console.warn;
  const seen = [];
  console.warn = (msg) => seen.push(msg);
  try {
    assert.equal(copy.t("ui.nope"), "ui.nope");
    assert.equal(copy.t("ui.nope"), "ui.nope");
  } finally {
    console.warn = warn;
  }
  assert.equal(seen.length, 1);
  assert.match(seen[0], /ui\.nope/);
});

test("a non-string entry is treated as missing, not a crash", () => {
  copy._setTable({ "ui.bad": null }, "en");
  const warn = console.warn;
  console.warn = () => {};
  try { assert.equal(copy.t("ui.bad"), "ui.bad"); } finally { console.warn = warn; }
});

test("loadCopy survives a failed fetch", async () => {
  const f = globalThis.fetch;
  const err = console.error;
  globalThis.fetch = async () => ({ ok: false, status: 404 });
  console.error = () => {};
  try {
    assert.deepEqual(await copy.loadCopy("en"), {});
    assert.equal(copy.lang(), "en");
  } finally {
    globalThis.fetch = f;
    console.error = err;
  }
});
