// U-WEB g711: the JS mu-law codec matches A's golden decode table and round-trips.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { ULAW_DECODE, decodeToFloat, encode, encodeSample, floatToInt16 } from "../static/js/g711.js";

const fixture = JSON.parse(readFileSync(new URL("../../tests/fixtures/audio/golden/ulaw_table.json", import.meta.url)));

test("decode table equals tests/fixtures/audio/golden/ulaw_table.json", () => {
  assert.equal(fixture.length, 256);
  assert.deepEqual(Array.from(ULAW_DECODE), fixture);
});

test("encode(decode(b)) == b for every byte (0x7F, negative zero, maps to 0xFF)", () => {
  for (let b = 0; b < 256; b++) {
    const want = b === 0x7f ? 0xff : b;
    assert.equal(encodeSample(ULAW_DECODE[b]), want, `byte ${b}`);
  }
});

test("silence, clipping and sign", () => {
  assert.equal(encodeSample(0), 0xff);
  assert.equal(encodeSample(32767), encodeSample(32635));
  assert.equal(encodeSample(-32768), encodeSample(-32635));
  assert.equal(encodeSample(32767), 0x80);
  assert.equal(encodeSample(-32768), 0x00);
});

test("decode(encode(s)) is monotone non-decreasing in s", () => {
  let prev = -Infinity;
  for (let s = -32768; s <= 32767; s++) {
    const v = ULAW_DECODE[encodeSample(s)];
    assert.ok(v >= prev, `s=${s}`);
    prev = v;
  }
});

test("array helpers", () => {
  const pcm = Int16Array.from([0, 1000, -1000, 32767]);
  const bytes = encode(pcm);
  assert.equal(bytes.length, 4);
  const f = decodeToFloat(bytes);
  assert.ok(Math.abs(f[1] - 1000 / 32768) < 0.01);
  assert.equal(floatToInt16(1.5), 32767);
  assert.equal(floatToInt16(-1.5), -32768);
  assert.equal(floatToInt16(0.5), 16384);
});
