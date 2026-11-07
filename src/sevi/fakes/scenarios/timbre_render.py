"""Fake Timbre `synthesize_wav` for offline calibration tests (no network, no credits)."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from sevi.contracts.enums import Lang


def fake_wav48(seed_text: str, *, fs: int = 48000, channels: int = 1, amp: float = 0.3) -> bytes:
    """150 ms silence, 450 ms tone (pitch from text) + seeded broadband noise, 150 ms silence."""
    f0 = 150.0 + (sum(map(ord, seed_text)) % 200)
    n_sil, n_tone = int(0.15 * fs), int(0.45 * fs)
    t = np.arange(n_tone) / fs
    rng = np.random.default_rng(sum(map(ord, seed_text)))
    tone = sum(np.sin(2 * np.pi * h * f0 * t) / h for h in range(1, 4)) / 1.9
    tone = amp * (tone + 1.0 * rng.standard_normal(n_tone))  # broadband: energy in all bands
    x = np.concatenate([np.zeros(n_sil), tone, np.zeros(n_sil)])
    if channels > 1:
        x = np.stack([x] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, fs, subtype="PCM_16", format="WAV")
    return buf.getvalue()


class FakeTimbre:
    """Matches contracts §5.2 ``TTSClient.synthesize_wav``; records every call."""

    def __init__(self, **wav_kw: object) -> None:
        self.calls: list[tuple[str, Lang, str, str, int]] = []
        self.wav_kw = wav_kw

    async def synthesize_wav(
        self, text: str, lang: Lang, voice: str, model: str, sample_rate_hz: int = 48000
    ) -> bytes:
        self.calls.append((text, lang, voice, model, sample_rate_hz))
        return fake_wav48(text, **self.wav_kw)  # type: ignore[arg-type]
