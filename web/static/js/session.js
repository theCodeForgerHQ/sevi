// Client-side session state machine (B spec §10.3) and the WebSocket client (contracts §9.4).
// `next()` is pure and unit-tested; `SessionClient` wires it to a socket, the player and the mic.
import * as wire from "./wire.js";

export const S = Object.freeze({
  BOOT: "BOOT", BACK_SOON: "BACK_SOON", LANDING: "LANDING", UNLOCK: "UNLOCK",
  ADMITTING: "ADMITTING", WAITING_ROOM: "WAITING_ROOM", RATE_LIMITED: "RATE_LIMITED",
  VOICE_UNAVAILABLE: "VOICE_UNAVAILABLE", QUEUED: "QUEUED", CONNECTING: "CONNECTING",
  CONSENT: "CONSENT", MIC: "MIC", INSTRUCTIONS: "INSTRUCTIONS", LISTEN: "LISTEN",
  ANSWER: "ANSWER", FEEDBACK: "FEEDBACK", RESULT: "RESULT", DONE: "DONE",
  INTERRUPTED: "INTERRUPTED", EXPIRED: "EXPIRED", ERROR: "ERROR",
});

/** States where the "I can't hear the numbers" button is visible. */
export const CANT_HEAR_STATES = new Set([S.CONSENT, S.MIC, S.INSTRUCTIONS, S.LISTEN, S.ANSWER, S.FEEDBACK]);
/** States in which an interruption ends the session (§10.7). */
export const LIVE_STATES = CANT_HEAR_STATES;
const TERMINAL = new Set([S.DONE, S.INTERRUPTED, S.EXPIRED, S.ERROR, S.RATE_LIMITED, S.BACK_SOON]);
const IN_SESSION = new Set([S.CONNECTING, ...CANT_HEAR_STATES]);

/**
 * Pure transition. Unknown or out-of-place events leave the state unchanged.
 * @param {string} state @param {{type: string, [k: string]: any}} ev
 * @returns {string}
 */
export function next(state, ev) {
  if (ev.type === "restart") return S.LANDING;
  if (TERMINAL.has(state) && state !== S.DONE) return state;
  switch (ev.type) {
    case "boot_ok": return state === S.BOOT ? S.LANDING : state;
    case "boot_down": return state === S.BOOT ? S.BACK_SOON : state;
    case "start": return state === S.LANDING || state === S.VOICE_UNAVAILABLE || state === S.QUEUED ? S.UNLOCK : state;
    case "unlocked": return state === S.UNLOCK ? S.ADMITTING : state;
    case "admit": {
      if (state !== S.ADMITTING && state !== S.WAITING_ROOM && state !== S.QUEUED) return state;
      if (ev.status === 201) return ev.body.status === "QUEUED" ? S.QUEUED : S.CONNECTING;
      if (ev.status === 503) return S.WAITING_ROOM;
      if (ev.status === 429) return S.RATE_LIMITED;
      if (ev.status === 409) return S.VOICE_UNAVAILABLE;
      return S.ERROR;
    }
    case "queue_admitted": return state === S.QUEUED ? S.CONNECTING : state;
    case "ui": return onUi(state, ev.msg);
    case "close": return onClose(state, ev.code, ev.sawClosed);
    case "interrupted": return IN_SESSION.has(state) ? S.INTERRUPTED : state;
    default: return state;
  }
}

function onUi(state, m) {
  if (state === S.DONE) return state;
  switch (m.type) {
    case "consent": return S.CONSENT;
    case "mic": return m.start ? S.MIC : state;
    case "phase":
      if (m.state === "INSTRUCTIONS") return S.INSTRUCTIONS;
      if (m.state === "TEST") return S.LISTEN;
      return state;
    case "keypad":
      if (state === S.RESULT) return state;
      return m.enabled ? S.ANSWER : S.LISTEN;
    case "feedback": return S.FEEDBACK;
    case "result": return S.RESULT;
    case "error": return S.ERROR;
    case "closed": return S.DONE;
    default: return state; // explanation, mode, notice, pong
  }
}

function onClose(state, code, sawClosed) {
  if (code === wire.CLOSE.TOKEN_INVALID) return S.EXPIRED;
  if (code === wire.CLOSE.RATE_LIMITED) return S.RATE_LIMITED;
  if (code === wire.CLOSE.PROTOCOL || code === wire.CLOSE.STATE_CONFLICT || code === wire.CLOSE.INTERNAL) return S.ERROR;
  if (state === S.DONE || state === S.ERROR) return state;
  if (sawClosed || state === S.RESULT) return S.DONE;
  return S.INTERRUPTED;
}

export const PING_MS = 15000;
export const PONG_TIMEOUT_MS = 10000;
export const BUFFERED_SLOW = 262144;

/**
 * WebSocket session client. Dependencies are injected so tests can run it in node.
 * opts: { wsUrl, sessionId, token, WebSocket, player?, onState, onUi, onBanner, setTimer?, clearTimer? }
 */
/** Mirrors `capture.t_answer_ms` (settings.py); tests/unit/b/test_web.py keeps them equal. */
export const ANSWER_MS = 8000;
export const AUTO_SUBMIT_MARGIN_MS = 600;

export class SessionClient {
  constructor(opts) {
    this.o = opts;
    this.state = S.CONNECTING;
    this.seq = new wire.Sequencer();
    this.sawClosed = false;
    this.presses = 0;
    this.digits = [];
    this.autoTimer = null;
    this.setTimeout = opts.setTimeout;
    this.clearTimeout = opts.clearTimeout;
    this.hearAgainUsed = false;
    this.keypadOpen = false;
    this.setTimer = opts.setTimer || ((f, ms) => setInterval(f, ms));
    this.clearTimer = opts.clearTimer || ((h) => clearInterval(h));
  }

  _go(ev) {
    const prev = this.state;
    this.state = next(this.state, ev);
    if (this.state !== prev && this.o.onState) this.o.onState(this.state, prev);
  }

  send(msg) {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify(msg));
    if (this.ws.bufferedAmount > BUFFERED_SLOW && this.o.onBanner) this.o.onBanner("ui.test.connection_slow");
    return true;
  }

  open() {
    const ws = new this.o.WebSocket(this.o.wsUrl);
    this.ws = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify(wire.connected()));
      this.send(wire.start(this.seq, this.o.sessionId, this.o.token));
      this.lastPong = Date.now();
      this.pinger = this.setTimer(() => this._ping(), PING_MS);
    };
    ws.onmessage = (e) => this._onMessage(e.data);
    ws.onclose = (e) => {
      this.clearTimer(this.pinger);
      this._cancelAuto();
      if (this.o.player) this.o.player.clear();
      if (this.o.mic) this.o.mic.stop();
      this._go({ type: "close", code: e.code, sawClosed: this.sawClosed });
    };
  }

  _ping() {
    if (Date.now() - this.lastPong > PING_MS + PONG_TIMEOUT_MS && this.o.onBanner) this.o.onBanner("ui.test.connection_slow");
    this.send(wire.ui(this.seq, "ping"));
  }

  _onMessage(text) {
    let m;
    try { m = wire.parseServer(text); } catch { return; } // the server tolerates nothing silently; we do
    if (m.event === "media") { if (this.o.player) this.o.player.enqueue(wire.b64ToBytes(m.media.payload)); return; }
    if (m.event === "mark") {
      const name = m.mark.name;
      if (this.o.player) this.o.player.mark(name, () => this.send(wire.mark(this.seq, name)));
      else this.send(wire.mark(this.seq, name));
      return;
    }
    if (m.event === "clear") {
      if (this.o.player) this.o.player.clear((name) => this.send(wire.mark(this.seq, name)));
      return;
    }
    const k = m.kj;
    if (k.type === "pong") { this.lastPong = Date.now(); return; }
    if (k.type === "closed") this.sawClosed = true;
    if (k.type === "phase") { this.hearAgainUsed = false; }
    if (k.type === "keypad") {
      this._cancelAuto();
      this.keypadOpen = k.enabled;
      if (k.enabled) {
        this.digits = [];
        this.presses = 0;
        // Held digits are sent just before the server's T_ANSWER (capture.t_answer_ms) closes
        // the window, so a typed-but-unconfirmed answer still counts.
        this.autoTimer = (this.setTimeout || setTimeout)(() => {
          this.autoTimer = null;
          if (this.keypadOpen && this.digits.length) { this.confirm(); if (this.o.onAutoSubmit) this.o.onAutoSubmit(); }
        }, ANSWER_MS - AUTO_SUBMIT_MARGIN_MS);
      }
    }
    if (k.type === "mic" && !k.start && this.o.mic) this.o.mic.stop();
    if (k.type === "mode" && k.mode === "keypad" && this.o.mic) this.o.mic.stop();
    if (this.o.onUi) this.o.onUi(k); // payload first, so the new state renders with its data
    this._go({ type: "ui", msg: k });
  }

  /** Keypad digit 1-9, held on the page until confirm (or the auto-submit just before the
   * server's answer window closes). Returns the number of digits entered so far. */
  press(digit) {
    if (!this.keypadOpen || this.digits.length >= 3) return this.digits.length;
    if (!/^[1-9]$/.test(digit)) return this.digits.length;
    this.digits.push(digit);
    this.presses = this.digits.length;
    return this.presses;
  }

  /** Remove the last held digit. Returns the count left. */
  del() {
    if (!this.keypadOpen) return this.digits.length;
    this.digits.pop();
    this.presses = this.digits.length;
    return this.presses;
  }

  /** Send the held digits as DTMF and close the keypad for this window. False if nothing held. */
  confirm() {
    if (!this.keypadOpen || this.digits.length === 0) return false;
    this._cancelAuto();
    for (const d of this.digits) this.send(wire.dtmf(this.seq, d));
    this.keypadOpen = false;
    return true;
  }

  _cancelAuto() {
    if (this.autoTimer != null) { (this.clearTimeout || clearTimeout)(this.autoTimer); this.autoTimer = null; }
  }

  hearAgain() {
    if (this.hearAgainUsed) return false;
    this.hearAgainUsed = true;
    return this.send(wire.dtmf(this.seq, "*"));
  }

  repeatInstructions() { return this.send(wire.dtmf(this.seq, "0")); }
  consent(agree) { return this.send(wire.ui(this.seq, agree ? "consent_given" : "consent_declined")); }
  cantHear() { return this.send(wire.ui(this.seq, "cant_hear_confirmed")); }
  micDenied() { return this.send(wire.ui(this.seq, "mic_denied")); }
  micLost() { return this.send(wire.ui(this.seq, "mic_lost")); }
  micSettings(s) { return this.send(wire.ui(this.seq, "mic_settings", s)); }
  media(frameIndex, tsMs, bytes) { return this.send(wire.media(this.seq, frameIndex, tsMs, wire.bytesToB64(bytes))); }

  /** Interruption (§10.7): send stop, close 1000, go INTERRUPTED. */
  interrupt() {
    if (!LIVE_STATES.has(this.state) && this.state !== S.CONNECTING) return;
    this.send(wire.stop(this.seq));
    this._go({ type: "interrupted" });
    if (this.ws) this.ws.close(1000);
  }
}
