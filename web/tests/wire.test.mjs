// Browser WS messages match contracts §9.4 byte for byte (field names, types, order of counters).
import assert from "node:assert/strict";
import { test } from "node:test";
import * as wire from "../static/js/wire.js";

const SID = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b";

test("connected", () => {
  assert.deepEqual(wire.connected(), { event: "connected", protocol: "KJWeb", version: "1.0.0" });
});

test("start is sequence 1 with streamSid WB + 12 hex", () => {
  const seq = new wire.Sequencer();
  assert.deepEqual(wire.start(seq, SID, "tok"), {
    event: "start", sequenceNumber: "1",
    start: {
      streamSid: "WB01933b6e8f1a", callSid: null,
      customParameters: { session_id: SID, token: "tok" },
      mediaFormat: { encoding: "audio/x-mulaw", sampleRate: 8000, channels: 1 },
    },
  });
  assert.equal(wire.mark(seq, "s009r0").sequenceNumber, "2");
});

test("media chunk is frameIndex+1, timestamp is ms, both strings", () => {
  const seq = new wire.Sequencer();
  seq.n = 41;
  assert.deepEqual(wire.media(seq, 40, 820, "AAA="), {
    event: "media", sequenceNumber: "42",
    media: { track: "inbound", chunk: "41", timestamp: "820", payload: "AAA=" },
  });
});

test("dtmf, mark, kj, stop", () => {
  const seq = new wire.Sequencer();
  assert.deepEqual(wire.mark(seq, "s009r0"), { event: "mark", sequenceNumber: "1", mark: { name: "s009r0" } });
  assert.deepEqual(wire.dtmf(seq, "4"), { event: "dtmf", sequenceNumber: "2", dtmf: { track: "inbound", digit: "4" } });
  assert.deepEqual(wire.ui(seq, "consent_given"), { event: "kj", sequenceNumber: "3", kj: { type: "consent_given" } });
  assert.deepEqual(wire.stop(seq), { event: "stop", sequenceNumber: "4" });
  assert.throws(() => wire.dtmf(seq, "#"));
  assert.throws(() => wire.ui(seq, "hello"));
});

test("mic_settings carries ec/ns/agc/sample_rate_hz", () => {
  const m = wire.ui(new wire.Sequencer(), "mic_settings", { ec: 0, ns: false, agc: true, sample_rate_hz: 48000.0 });
  assert.deepEqual(m.kj, { type: "mic_settings", ec: false, ns: false, agc: true, sample_rate_hz: 48000 });
});

test("every BrowserUiEvent type is buildable", () => {
  for (const ty of wire.UI_EVENTS) assert.equal(wire.ui(new wire.Sequencer(), ty).kj.type, ty);
  assert.deepEqual([...wire.UI_EVENTS].sort(), ["cant_hear_confirmed", "consent_declined", "consent_given", "mic_denied", "mic_lost", "mic_settings", "ping"]);
});

test("parseServer accepts every §9.4 server message", () => {
  const ss = { streamSid: "WB01933b6e8f1a" };
  const ok = [
    { event: "media", ...ss, media: { payload: "AAA=" } },
    { event: "mark", ...ss, mark: { name: "s009r0" } },
    { event: "clear", ...ss },
    { event: "kj", ...ss, kj: { type: "phase", state: "TEST", kind: "SCORED", scored_index: 4, total: 24 } },
    { event: "kj", ...ss, kj: { type: "phase", state: "INSTRUCTIONS", kind: null, scored_index: null, total: 24 } },
    { event: "kj", ...ss, kj: { type: "consent", script: "PUBLIC_V1", text: "x" } },
    { event: "kj", ...ss, kj: { type: "keypad", enabled: true } },
    { event: "kj", ...ss, kj: { type: "mode", mode: "keypad" } },
    { event: "kj", ...ss, kj: { type: "mic", start: true } },
    { event: "kj", ...ss, kj: { type: "feedback", correct: false } },
    { event: "kj", ...ss, kj: { type: "result", band: "PASS", wording_class: "W_PASS", page_text: "t", srt_display: null, report_url: "" } },
    { event: "kj", ...ss, kj: { type: "explanation", text: "e" } },
    { event: "kj", ...ss, kj: { type: "error", code: "INTERNAL", message: "m" } },
    { event: "kj", ...ss, kj: { type: "closed" } },
    { event: "kj", ...ss, kj: { type: "pong" } },
    { event: "kj", ...ss, kj: { type: "notice", key: "speaker_echo" } },
  ];
  for (const m of ok) assert.deepEqual(wire.parseServer(JSON.stringify(m)), m);
  assert.deepEqual([...wire.SERVER_UI_TYPES].sort(), ["closed", "consent", "error", "explanation", "feedback", "keypad", "mic", "mode", "notice", "phase", "pong", "result"]);
});

test("parseServer rejects anything outside §9.4", () => {
  const bad = [
    { event: "nope" },
    { event: "kj", kj: { type: "phase", state: "TEST", kind: "SCORED", scored_index: 25, total: 24 } },
    { event: "kj", kj: { type: "keypad", enabled: "yes" } },
    { event: "kj", kj: { type: "notice", key: "other" } },
    { event: "kj", kj: { type: "explanation", text: "x".repeat(301) } },
    { event: "media", media: {} },
  ];
  for (const m of bad) assert.throws(() => wire.parseServer(JSON.stringify(m)), JSON.stringify(m));
});

test("b64 round trip", () => {
  globalThis.btoa ??= (s) => Buffer.from(s, "binary").toString("base64");
  const b = Uint8Array.from({ length: 160 }, (_, i) => (i * 37) & 0xff);
  assert.deepEqual(wire.b64ToBytes(wire.bytesToB64(b)), b);
});
