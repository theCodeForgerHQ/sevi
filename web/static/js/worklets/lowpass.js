// 127-tap Hamming-windowed sinc low-pass, fc = 3600 Hz at the context rate, evaluated at
// fractional input positions via a 128-phase polyphase table with linear interpolation
// between phases (contracts §9.4, B spec §10.4.3). Pure; shared by the worklet and tests.

export const TAPS = 127;
export const HALF = 63;
export const PHASES = 128;
export const CUTOFF_HZ = 3600;
export const OUT_RATE = 8000;

/** Filter value at continuous offset x (input samples) from the centre, unnormalised. */
function kernel(x, fcNorm) {
  // fcNorm = fc / fs (cycles per input sample)
  const sinc = x === 0 ? 2 * fcNorm : Math.sin(2 * Math.PI * fcNorm * x) / (Math.PI * x);
  const w = Math.abs(x) > HALF ? 0 : 0.54 + 0.46 * Math.cos((Math.PI * x) / (HALF + 1));
  return sinc * w;
}

/**
 * Build the polyphase table: PHASES + 1 rows (row PHASES == phase 1.0 for interpolation),
 * each TAPS long, row p holds taps for fractional offset frac = p / PHASES, normalised to DC gain 1.
 * Tap k (0..126) multiplies input sample at index floor(t) + (k - HALF), weight h(k - HALF - frac).
 * @param {number} sampleRate
 */
export function buildTable(sampleRate) {
  const fcNorm = CUTOFF_HZ / sampleRate;
  const table = [];
  for (let p = 0; p <= PHASES; p++) {
    const frac = p / PHASES;
    const row = new Float32Array(TAPS);
    let sum = 0;
    for (let k = 0; k < TAPS; k++) {
      const v = kernel(k - HALF - frac, fcNorm);
      row[k] = v;
      sum += v;
    }
    for (let k = 0; k < TAPS; k++) row[k] /= sum;
    table.push(row);
  }
  return table;
}

/**
 * Taps for an arbitrary fractional offset (linear interpolation between phases).
 * @param {Float32Array[]} table @param {number} frac in [0, 1) @param {Float32Array} out
 */
export function tapsAt(table, frac, out) {
  const pos = frac * PHASES;
  const p = Math.floor(pos);
  const a = pos - p;
  const r0 = table[p];
  const r1 = table[p + 1];
  for (let k = 0; k < TAPS; k++) out[k] = r0[k] + a * (r1[k] - r0[k]);
  return out;
}

/**
 * Stateful downsampler: push input blocks, get 8 kHz float samples.
 * Ring holds the last 512 input samples (>= 384 per spec).
 */
export class Downsampler {
  constructor(sampleRate) {
    this.rate = sampleRate;
    this.step = sampleRate / OUT_RATE;
    this.table = buildTable(sampleRate);
    this.ringSize = 512;
    this.ring = new Float32Array(this.ringSize);
    this.written = 0; // total input samples seen
    this.nextT = HALF; // input position of the next output (needs HALF samples of history)
    this.taps = new Float32Array(TAPS);
  }

  /** @param {Float32Array} input @returns {number[]} */
  push(input) {
    const out = [];
    for (let i = 0; i < input.length; i++) {
      this.ring[this.written % this.ringSize] = input[i];
      this.written++;
      // produce every output whose right-most tap is now available
      while (Math.floor(this.nextT) + HALF < this.written) {
        const base = Math.floor(this.nextT);
        tapsAt(this.table, this.nextT - base, this.taps);
        let y = 0;
        for (let k = 0; k < TAPS; k++) {
          const idx = base + k - HALF;
          y += this.taps[k] * this.ring[((idx % this.ringSize) + this.ringSize) % this.ringSize];
        }
        out.push(y);
        this.nextT += this.step;
      }
    }
    return out;
  }
}

/** Magnitude response in dB of the phase-0 filter at frequency f (for tests). */
const cache = new Map();
export function responseDb(sampleRate, f) {
  if (!cache.has(sampleRate)) cache.set(sampleRate, buildTable(sampleRate)[0]);
  const row = cache.get(sampleRate);
  let re = 0;
  let im = 0;
  for (let k = 0; k < TAPS; k++) {
    const w = (2 * Math.PI * f * (k - HALF)) / sampleRate;
    re += row[k] * Math.cos(w);
    im -= row[k] * Math.sin(w);
  }
  return 20 * Math.log10(Math.hypot(re, im));
}
