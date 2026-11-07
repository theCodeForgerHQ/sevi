// Browser WebSocket messages, exactly per contracts §9.4 (`sevi.contracts.wire`).
// Builders for client -> server frames and a strict parser for server -> client frames.

export const MEDIA_FORMAT = Object.freeze({ encoding: "audio/x-mulaw", sampleRate: 8000, channels: 1 });

/** Page-side UI events (BrowserUiEvent.type). */
export const UI_EVENTS = Object.freeze([
  "consent_given", "consent_declined", "cant_hear_confirmed", "ping",
  "mic_denied", "mic_lost", "mic_settings",
]);

export const SESSION_STATES = Object.freeze([
  "CREATED", "QUEUED", "DIALING", "AWAIT_START", "CONSENT", "INSTRUCTIONS",
  "TEST", "RESULT", "CLOSING", "ENDED",
]);
export const PRESENTATION_KINDS = Object.freeze(["VOLUME", "PRACTICE", "SCORED", "LOUD"]);
export const CLOSE = Object.freeze({
  NORMAL: 1000, GOING_AWAY: 1001, ABNORMAL: 1006,
  PROTOCOL: 4400, TOKEN_INVALID: 4401, STATE_CONFLICT: 4409, RATE_LIMITED: 4429, INTERNAL: 4500,
});

/** streamSid = "WB" + sid without dashes, first 12 hex chars. */
export function streamSidFor(sessionId) {
  return "WB" + sessionId.replace(/-/g, "").slice(0, 12);
}

/** sequenceNumber: string counter starting at "1" for `start`, +1 per message after. */
export class Sequencer {
  constructor() { this.n = 0; }
  next() { this.n += 1; return String(this.n); }
}

export function connected() {
  return { event: "connected", protocol: "KJWeb", version: "1.0.0" };
}

export function start(seq, sessionId, token) {
  return {
    event: "start",
    sequenceNumber: seq.next(),
    start: {
      streamSid: streamSidFor(sessionId),
      callSid: null,
      customParameters: { session_id: sessionId, token },
      mediaFormat: { ...MEDIA_FORMAT },
    },
  };
}

/** @param {Sequencer} seq @param {number} frameIndex 0-based @param {number} tsMs @param {string} payloadB64 */
export function media(seq, frameIndex, tsMs, payloadB64) {
  return {
    event: "media",
    sequenceNumber: seq.next(),
    media: { track: "inbound", chunk: String(frameIndex + 1), timestamp: String(tsMs), payload: payloadB64 },
  };
}

export function mark(seq, name) {
  return { event: "mark", sequenceNumber: seq.next(), mark: { name } };
}

const DIGITS = new Set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]);
export function dtmf(seq, digit) {
  if (!DIGITS.has(digit)) throw new Error(`dtmf: bad digit ${digit}`);
  return { event: "dtmf", sequenceNumber: seq.next(), dtmf: { track: "inbound", digit } };
}

/** @param {string} type one of UI_EVENTS @param {object} [extra] mic_settings fields */
export function ui(seq, type, extra = {}) {
  if (!UI_EVENTS.includes(type)) throw new Error(`ui: bad type ${type}`);
  const body = { type };
  if (type === "mic_settings") {
    body.ec = Boolean(extra.ec);
    body.ns = Boolean(extra.ns);
    body.agc = Boolean(extra.agc);
    body.sample_rate_hz = Math.round(extra.sample_rate_hz);
  }
  return { event: "kj", sequenceNumber: seq.next(), kj: body };
}

export function stop(seq) {
  return { event: "stop", sequenceNumber: seq.next() };
}

// ---------- server -> client ----------

const isStr = (v) => typeof v === "string";
const isBool = (v) => typeof v === "boolean";
const isInt = (v) => Number.isInteger(v);

const UI_SCHEMAS = {
  consent: (m) => ["PUBLIC_V1", "STUDY_V1"].includes(m.script) && isStr(m.text),
  phase: (m) =>
    SESSION_STATES.includes(m.state) &&
    (m.kind === null || PRESENTATION_KINDS.includes(m.kind)) &&
    (m.scored_index === null || m.scored_index === undefined || (isInt(m.scored_index) && m.scored_index >= 1 && m.scored_index <= 24)) &&
    (m.total === undefined || m.total === 24),
  keypad: (m) => isBool(m.enabled),
  mode: (m) => m.mode === "keypad" || m.mode === "voice",
  mic: (m) => isBool(m.start),
  feedback: (m) => isBool(m.correct),
  result: (m) => isStr(m.band) && isStr(m.wording_class) && isStr(m.page_text) &&
    (m.srt_display === null || isStr(m.srt_display)) && isStr(m.report_url),
  explanation: (m) => isStr(m.text) && m.text.length <= 300,
  error: (m) => isStr(m.code) && isStr(m.message),
  closed: () => true,
  pong: () => true,
  notice: (m) => m.key === "speaker_echo",
};
export const SERVER_UI_TYPES = Object.freeze(Object.keys(UI_SCHEMAS));

/**
 * Parse one server text frame. Returns {event, ...} or throws on anything outside §9.4.
 * @param {string} text
 */
export function parseServer(text) {
  const m = JSON.parse(text);
  if (!m || typeof m !== "object" || !isStr(m.event)) throw new Error("wire: no event");
  switch (m.event) {
    case "media":
      if (!m.media || !isStr(m.media.payload)) throw new Error("wire: media without payload");
      return m;
    case "mark":
      if (!m.mark || !isStr(m.mark.name)) throw new Error("wire: mark without name");
      return m;
    case "clear":
      return m;
    case "kj": {
      const k = m.kj;
      if (!k || !UI_SCHEMAS[k.type]) throw new Error(`wire: unknown ui type ${k && k.type}`);
      if (!UI_SCHEMAS[k.type](k)) throw new Error(`wire: bad ${k.type} fields`);
      return m;
    }
    default:
      throw new Error(`wire: unknown event ${m.event}`);
  }
}

/** base64 of a Uint8Array without TextEncoder tricks (works in worker and page). */
export function bytesToB64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

export function b64ToBytes(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
