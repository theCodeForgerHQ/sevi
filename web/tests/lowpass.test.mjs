// U-WEB-02 (browser half): the 127-tap capture low-pass meets the §10.4.3 response.
import assert from "node:assert/strict";
import { test } from "node:test";
import { Downsampler, PHASES, TAPS, buildTable, responseDb } from "../static/js/worklets/lowpass.js";

for (const sr of [48000, 44100]) {
  test(`response at ${sr} Hz`, () => {
    assert.ok(Math.abs(responseDb(sr, 1000)) < 0.05);
    assert.ok(responseDb(sr, 3000) > -0.1, "3.0 kHz passband");
    const r34 = responseDb(sr, 3400);
    assert.ok(r34 < -1.5 && r34 > -3, `3.4 kHz ${r34}`);
    assert.ok(responseDb(sr, 4000) < -20, "4.0 kHz");
    for (let f = 4400; f < sr / 2; f += 25) assert.ok(responseDb(sr, f) <= -55, `${f} Hz`);
  });
}

test("polyphase table shape and unit DC gain per phase", () => {
  const t = buildTable(48000);
  assert.equal(TAPS, 127);
  assert.equal(t.length, PHASES + 1);
  for (const row of t) assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-5);
});

test("downsampler output rate is 8 kHz and passes a 1 kHz tone", () => {
  const sr = 48000;
  const ds = new Downsampler(sr);
  const n = sr; // 1 s
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / sr);
  const y = [];
  for (let i = 0; i < n; i += 128) y.push(...ds.push(x.subarray(i, i + 128)));
  // 63-sample warm-up + 63-sample look-ahead at 48 kHz = ~21 output samples of latency
  assert.ok(y.length <= 8000 && y.length >= 8000 - 32, `got ${y.length}`);
  const tail = y.slice(200, 200 + 7200); // whole periods of 1 kHz at 8 kHz
  const rms = Math.sqrt(tail.reduce((a, v) => a + v * v, 0) / tail.length);
  assert.ok(Math.abs(rms - 0.5 / Math.SQRT2) < 0.005, `rms ${rms}`);
});
