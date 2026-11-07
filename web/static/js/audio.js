// AudioContext lifecycle, native playback of 8 kHz mu-law clips with mark echo, and mic capture
// through the `sevi-capture` worklet (B spec §10.4, §10.6). Browser-only.
import { decodeToFloat } from "./g711.js";

/**
 * Create and unlock the AudioContext. Must be called synchronously inside the Start tap.
 * No sampleRate option (Firefox rejects connecting a mic to a context at another rate).
 */
export function unlock({ voice }) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx({ latencyHint: "interactive" });
  if (navigator.audioSession) navigator.audioSession.type = voice ? "play-and-record" : "playback";
  ctx.resume();
  const silent = ctx.createBufferSource();
  silent.buffer = ctx.createBuffer(1, Math.round(ctx.sampleRate / 10), ctx.sampleRate);
  silent.connect(ctx.destination);
  silent.start();
  return ctx;
}

export function outputLatency(ctx) {
  return Math.min(0.4, ctx.outputLatency || ctx.baseLatency || 0);
}

/** Schedules clips back to back; a mark fires `onended` of a silent tail sized to output latency. */
export class Player {
  constructor(ctx) {
    this.ctx = ctx;
    this.lastEnd = 0;
    this.sources = new Set();
    this.pendingMarks = new Map(); // name -> send fn
    this.clipSinceMark = false;
  }

  enqueue(bytes) {
    const ctx = this.ctx;
    const data = decodeToFloat(bytes);
    const buf = ctx.createBuffer(1, data.length, 8000);
    buf.copyToChannel(data, 0);
    this._schedule(buf);
    this.clipSinceMark = true;
  }

  _schedule(buf, onended) {
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const t0 = Math.max(this.ctx.currentTime + 0.02, this.lastEnd);
    src.start(t0);
    this.lastEnd = t0 + buf.length / buf.sampleRate;
    this.sources.add(src);
    src.onended = () => { this.sources.delete(src); if (onended) onended(); };
  }

  /** Echo `name` once everything queued before it has played (plus output latency). */
  mark(name, send) {
    if (!this.clipSinceMark) { send(); return; }
    this.clipSinceMark = false;
    const n = Math.max(1, Math.round(outputLatency(this.ctx) * 8000));
    this.pendingMarks.set(name, send);
    this._schedule(this.ctx.createBuffer(1, n, 8000), () => {
      const fn = this.pendingMarks.get(name);
      this.pendingMarks.delete(name);
      if (fn) fn();
    });
  }

  /** `clear`: stop everything and post every pending mark at once. */
  clear() {
    for (const s of this.sources) { s.onended = null; try { s.stop(); } catch { /* not started */ } }
    this.sources.clear();
    const marks = [...this.pendingMarks.values()];
    this.pendingMarks.clear();
    for (const fn of marks) fn();
    this.lastEnd = this.ctx.currentTime;
    this.clipSinceMark = false;
  }
}

export const MIC_MUTE_LOST_MS = 2000;

/**
 * Microphone capture. Call `start()` only after ui `mic {start:true}` (i.e. after consent).
 * hooks: { onFrame(frameIndex, tsMs, bytes), onSettings(s), onDenied(), onLost() }
 */
export class Mic {
  constructor(ctx, { earphones, hooks }) {
    this.ctx = ctx;
    this.earphones = earphones;
    this.h = hooks;
    this.stream = null;
    this.frameIndex = 0;
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: !this.earphones, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
        video: false,
      });
    } catch {
      // NotAllowed, NotFound, NotReadable, Overconstrained, Security: all fall back to keypad.
      this.h.onDenied();
      return false;
    }
    const track = this.stream.getAudioTracks()[0];
    const s = track.getSettings();
    this.h.onSettings({
      ec: !!s.echoCancellation, ns: !!s.noiseSuppression, agc: !!s.autoGainControl,
      sample_rate_hz: s.sampleRate || this.ctx.sampleRate,
    });
    track.onended = () => this.h.onLost();
    let muteTimer = null;
    track.onmute = () => { muteTimer = setTimeout(() => this.h.onLost(), MIC_MUTE_LOST_MS); };
    track.onunmute = () => { clearTimeout(muteTimer); };

    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "sevi-capture");
    this.node.port.onmessage = (e) => {
      const { bytes, ts_ms } = e.data;
      this.h.onFrame(this.frameIndex++, ts_ms, bytes);
    };
    const sink = this.ctx.createGain();
    sink.gain.value = 0; // zero-gain sink keeps Safari pulling the node
    src.connect(this.node).connect(sink).connect(this.ctx.destination);
    return true;
  }

  stop() {
    if (this.stream) for (const t of this.stream.getTracks()) { t.onended = null; t.stop(); }
    this.stream = null;
    if (this.node) { this.node.port.onmessage = null; this.node.disconnect(); this.node = null; }
  }
}

export async function loadCaptureWorklet(ctx) {
  await ctx.audioWorklet.addModule(new URL("./worklets/capture.js", import.meta.url));
}
