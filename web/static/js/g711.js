// G.711 mu-law (ITU-T segment algorithm, bias 0x84, clip 32635).
// Mirrors src/sevi/core/dsp.py `_ulaw_encode_sample` / `_ulaw_decode_byte` bit for bit.
// Checked against tests/fixtures/audio/golden/ulaw_table.json by web/tests/g711.test.mjs.

export const ULAW_BIAS = 0x84;
export const ULAW_CLIP = 32635;

function decodeByte(b) {
  const u = ~b & 0xff;
  const e = (u >> 4) & 7;
  const mant = u & 0x0f;
  const v = (((mant << 3) + ULAW_BIAS) << e) - ULAW_BIAS;
  return u & 0x80 ? -v : v;
}

/** @type {Int16Array} byte -> int16 sample */
export const ULAW_DECODE = (() => {
  const t = new Int16Array(256);
  for (let b = 0; b < 256; b++) t[b] = decodeByte(b);
  return t;
})();

/** @param {number} s int16 sample @returns {number} mu-law byte */
export function encodeSample(s) {
  const sign = s < 0 ? 0x80 : 0;
  const m = Math.min(Math.abs(s), ULAW_CLIP) + ULAW_BIAS;
  const e = 31 - Math.clz32(m >> 7); // floor(log2(m >> 7)); m >= 132
  const mant = (m >> (e + 3)) & 0x0f;
  return ~(sign | (e << 4) | mant) & 0xff;
}

/** @param {Int16Array} pcm @returns {Uint8Array} */
export function encode(pcm) {
  const out = new Uint8Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = encodeSample(pcm[i]);
  return out;
}

/** mu-law bytes -> Float32 in [-1, 1) at the same rate. @param {Uint8Array} bytes */
export function decodeToFloat(bytes) {
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = ULAW_DECODE[bytes[i]] / 32768;
  return out;
}

/** Float in [-1, 1] -> int16 per spec: clamp(round(y * 32768), -32768, 32767). */
export function floatToInt16(y) {
  const v = Math.round(y * 32768);
  return v < -32768 ? -32768 : v > 32767 ? 32767 : v;
}
