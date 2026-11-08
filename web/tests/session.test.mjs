// Client FSM (B spec §10.3) and SessionClient behaviour against a fake socket.
import assert from "node:assert/strict";
import { test } from "node:test";
import { S, SessionClient, next } from "../static/js/session.js";

const ui = (msg) => ({ type: "ui", msg });

test("boot and admission transitions", () => {
  assert.equal(next(S.BOOT, { type: "boot_ok" }), S.LANDING);
  assert.equal(next(S.BOOT, { type: "boot_down" }), S.BACK_SOON);
  assert.equal(next(S.LANDING, { type: "start" }), S.UNLOCK);
  assert.equal(next(S.UNLOCK, { type: "unlocked" }), S.ADMITTING);
  const adm = (status, body = {}) => next(S.ADMITTING, { type: "admit", status, body });
  assert.equal(adm(201, { status: "ADMITTED" }), S.CONNECTING);
  assert.equal(adm(201, { status: "QUEUED" }), S.QUEUED);
  assert.equal(adm(503), S.WAITING_ROOM);
  assert.equal(adm(429), S.RATE_LIMITED);
  assert.equal(adm(409), S.VOICE_UNAVAILABLE);
  assert.equal(adm(500), S.ERROR);
  assert.equal(next(S.WAITING_ROOM, { type: "admit", status: 201, body: { status: "ADMITTED" } }), S.CONNECTING);
  assert.equal(next(S.QUEUED, { type: "queue_admitted" }), S.CONNECTING);
  assert.equal(next(S.VOICE_UNAVAILABLE, { type: "start" }), S.UNLOCK);
});

test("server ui messages drive the test states", () => {
  let s = S.CONNECTING;
  const steps = [
    [{ type: "consent", script: "PUBLIC_V1", text: "x" }, S.CONSENT],
    [{ type: "mic", start: true }, S.MIC],
    [{ type: "phase", state: "INSTRUCTIONS", kind: null }, S.INSTRUCTIONS],
    [{ type: "phase", state: "TEST", kind: "PRACTICE" }, S.LISTEN],
    [{ type: "keypad", enabled: true }, S.ANSWER],
    [{ type: "keypad", enabled: false }, S.LISTEN],
    [{ type: "feedback", correct: true }, S.FEEDBACK],
    [{ type: "phase", state: "TEST", kind: "SCORED", scored_index: 1 }, S.LISTEN],
    [{ type: "pong" }, S.LISTEN],
    [{ type: "notice", key: "speaker_echo" }, S.LISTEN],
    [{ type: "mode", mode: "keypad" }, S.LISTEN],
    [{ type: "result", band: "PASS" }, S.RESULT],
    [{ type: "mic", start: false }, S.RESULT],
    [{ type: "keypad", enabled: false }, S.RESULT],
    [{ type: "explanation", text: "e" }, S.RESULT],
    [{ type: "closed" }, S.DONE],
  ];
  for (const [msg, want] of steps) { s = next(s, ui(msg)); assert.equal(s, want, JSON.stringify(msg)); }
  assert.equal(next(s, { type: "close", code: 1000, sawClosed: true }), S.DONE);
});

test("close codes", () => {
  assert.equal(next(S.LISTEN, { type: "close", code: 4401 }), S.EXPIRED);
  assert.equal(next(S.CONNECTING, { type: "close", code: 4429 }), S.RATE_LIMITED);
  for (const c of [4400, 4409, 4500]) assert.equal(next(S.ANSWER, { type: "close", code: c }), S.ERROR);
  assert.equal(next(S.ANSWER, { type: "close", code: 1006 }), S.INTERRUPTED);
  assert.equal(next(S.ANSWER, { type: "close", code: 1001 }), S.INTERRUPTED);
  assert.equal(next(S.RESULT, { type: "close", code: 1006 }), S.DONE);
  assert.equal(next(S.ANSWER, ui({ type: "error", code: "INTERNAL", message: "m" })), S.ERROR);
  assert.equal(next(S.ERROR, { type: "close", code: 1000 }), S.ERROR);
});

test("interruption only in live states; terminal states are sticky until restart", () => {
  assert.equal(next(S.ANSWER, { type: "interrupted" }), S.INTERRUPTED);
  assert.equal(next(S.LANDING, { type: "interrupted" }), S.LANDING);
  assert.equal(next(S.INTERRUPTED, ui({ type: "consent", script: "PUBLIC_V1", text: "" })), S.INTERRUPTED);
  assert.equal(next(S.EXPIRED, { type: "restart" }), S.LANDING);
});

class FakeWS {
  constructor(url) { this.url = url; this.sent = []; this.readyState = 1; this.bufferedAmount = 0; FakeWS.last = this; }
  send(t) { this.sent.push(JSON.parse(t)); }
  close(code) { this.closed = code; this.onclose({ code }); }
  emit(kj) { this.onmessage({ data: JSON.stringify({ event: "kj", streamSid: "WB", kj }) }); }
}

function client(extra = {}) {
  const states = [];
  const c = new SessionClient({
    wsUrl: "wss://x/web/stream?sid=s", sessionId: "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b", token: "t",
    WebSocket: FakeWS, onState: (s) => states.push(s), setTimer: () => 0, clearTimer: () => {}, setTimeout: () => 0, clearTimeout: () => {}, ...extra,
  });
  c.open();
  FakeWS.last.onopen();
  return { c, ws: FakeWS.last, states };
}

test("open sends connected (no sequence) then start with sequence 1", () => {
  const { ws } = client();
  assert.equal(ws.sent[0].event, "connected");
  assert.equal(ws.sent[0].sequenceNumber, undefined);
  assert.equal(ws.sent[1].event, "start");
  assert.equal(ws.sent[1].sequenceNumber, "1");
});

test("keypad: digits held until confirm; delete; max three; sequence increments", () => {
  const { c, ws } = client();
  assert.equal(c.press("4"), 0, "closed keypad ignores presses");
  ws.emit({ type: "phase", state: "TEST", kind: "SCORED", scored_index: 1, total: 24 });
  ws.emit({ type: "keypad", enabled: true });
  c.press("4"); c.press("0"); c.press("2"); c.press("9"); c.press("7");
  assert.deepEqual(c.digits, ["4", "2", "9"], "non-digits ignored, max three");
  assert.equal(ws.sent.filter((m) => m.event === "dtmf").length, 0, "nothing sent before confirm");
  assert.equal(c.del(), 2);
  c.press("5");
  assert.equal(c.confirm(), true);
  const d = ws.sent.filter((m) => m.event === "dtmf").map((m) => m.dtmf.digit);
  assert.deepEqual(d, ["4", "2", "5"]);
  assert.equal(c.press("1"), 3, "keypad closed after confirm");
  assert.equal(c.confirm(), false);
  const seqs = ws.sent.slice(1).map((m) => Number(m.sequenceNumber));
  assert.deepEqual(seqs, seqs.map((_, i) => i + 1));
  ws.emit({ type: "keypad", enabled: true });
  assert.equal(c.press("1"), 1, "a new window resets the count");
});

test("keypad: held digits auto-submit just before the answer window closes", () => {
  let fire = null;
  const { c, ws } = client({ setTimeout: (f) => { fire = f; return 1; }, clearTimeout: () => { fire = null; } });
  ws.emit({ type: "keypad", enabled: true });
  c.press("3"); c.press("8");
  assert.ok(fire, "auto-submit armed");
  fire();
  assert.deepEqual(ws.sent.filter((m) => m.event === "dtmf").map((m) => m.dtmf.digit), ["3", "8"]);
});

test("hear again: once per presentation", () => {
  const { c, ws } = client();
  ws.emit({ type: "phase", state: "TEST", kind: "SCORED", scored_index: 1, total: 24 });
  assert.equal(c.hearAgain(), true);
  assert.equal(c.hearAgain(), false);
  ws.emit({ type: "phase", state: "TEST", kind: "SCORED", scored_index: 2, total: 24 });
  assert.equal(c.hearAgain(), true);
  assert.equal(ws.sent.filter((m) => m.event === "dtmf" && m.dtmf.digit === "*").length, 2);
});

test("marks echo through the player; without a player they echo at once", () => {
  const { ws } = client();
  ws.onmessage({ data: JSON.stringify({ event: "mark", streamSid: "WB", mark: { name: "s001r0" } }) });
  assert.deepEqual(ws.sent.at(-1).mark, { name: "s001r0" });
  const queued = [];
  const player = { enqueue() {}, mark: (n, send) => queued.push(send), clear() {} };
  const { ws: ws2 } = client({ player });
  ws2.onmessage({ data: JSON.stringify({ event: "mark", streamSid: "WB", mark: { name: "s002r0" } }) });
  assert.equal(ws2.sent.at(-1).event, "start");
  queued[0]();
  assert.deepEqual(ws2.sent.at(-1).mark, { name: "s002r0" });
});

test("consent, cant-hear and mic events", () => {
  const { c, ws } = client();
  c.consent(true); c.consent(false); c.cantHear(); c.micDenied(); c.micLost();
  c.micSettings({ ec: false, ns: false, agc: false, sample_rate_hz: 48000 });
  assert.deepEqual(ws.sent.slice(2).map((m) => m.kj.type), ["consent_given", "consent_declined", "cant_hear_confirmed", "mic_denied", "mic_lost", "mic_settings"]);
});

test("closed + 1000 ends in DONE; interrupt sends stop and closes 1000", () => {
  const a = client();
  a.ws.emit({ type: "result", band: "PASS", wording_class: "W_PASS", page_text: "p", srt_display: null, report_url: "" });
  a.ws.emit({ type: "closed" });
  a.ws.close(1000);
  assert.equal(a.c.state, S.DONE);

  const b = client();
  b.ws.emit({ type: "consent", script: "PUBLIC_V1", text: "x" });
  b.c.interrupt();
  assert.equal(b.ws.sent.at(-1).event, "stop");
  assert.equal(b.ws.closed, 1000);
  assert.equal(b.c.state, S.INTERRUPTED);
});

test("unparseable server frames are ignored", () => {
  const { c, ws } = client();
  ws.onmessage({ data: "not json" });
  ws.onmessage({ data: JSON.stringify({ event: "kj", kj: { type: "bogus" } }) });
  assert.equal(c.state, S.CONNECTING);
});
