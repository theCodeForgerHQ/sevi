// The only AudioWorkletProcessor: context-rate mic -> 127-tap low-pass -> 8 kHz -> mu-law,
// posted as 20 ms frames {type:"frame", bytes: Uint8Array(160), ts_ms} (B spec §10.4.3).
import { encodeSample, floatToInt16 } from "../g711.js";
import { Downsampler } from "./lowpass.js";

const FRAME = 160;

class SeviCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` is a global in AudioWorkletGlobalScope.
    this.ds = new Downsampler(sampleRate);
    this.frame = new Uint8Array(FRAME);
    this.fill = 0;
    this.frameIndex = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (const y of this.ds.push(ch)) {
      this.frame[this.fill++] = encodeSample(floatToInt16(y));
      if (this.fill === FRAME) {
        const bytes = this.frame;
        this.port.postMessage({ type: "frame", bytes, ts_ms: 20 * this.frameIndex }, [bytes.buffer]);
        this.frameIndex++;
        this.frame = new Uint8Array(FRAME);
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor("sevi-capture", SeviCapture);
