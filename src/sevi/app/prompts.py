"""Prompt assets and mark naming (B spec §7.12, contracts §6.4).

A prompt set is a directory ``<base>/<lang>/`` holding ``manifest.json`` (``PromptManifest``) and
one headerless 8 kHz mu-law file per ``PromptKey``. ``load_prompts`` verifies every file sha256
and the ``prompts_hash`` and refuses a set that lacks a required key. ``placeholder_prompts``
builds an in-memory silent set (voice ``"placeholder"``) for dev, test and demo; prod refuses it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from sevi.contracts.constants import MULAW_SILENCE_BYTE, SAMPLE_RATE_WIRE_HZ
from sevi.contracts.enums import Lang, PromptKey
from sevi.contracts.errors import BankError, ErrorCode
from sevi.contracts.stimulus import BankManifest, PromptManifest

MANIFEST_NAME: Final = "manifest.json"
PLACEHOLDER_VOICE: Final = "placeholder"
BYTES_PER_MS: Final = SAMPLE_RATE_WIRE_HZ // 1000  # 8 mu-law bytes per ms
# W_NOREPLY has no clip (contracts §6.5); it is not a PromptKey member at all.
NO_CLIP_WORDING: Final = frozenset[str]({"W_NOREPLY"})


# ---------- mark names (contracts §6.4) ----------
def mark_for_prompt(seq: int) -> str:
    return f"p{seq:04d}"


def mark_for_stimulus(presentation_id: int, replay: int) -> str:
    return f"s{presentation_id:03d}r{replay}"


def mark_for_feedback(presentation_id: int) -> str:
    return f"f{presentation_id:03d}"


def mark_for_playlist(index: int) -> str:
    return f"z{index:02d}"


def digit_key(d: int) -> PromptKey:
    return PromptKey(f"DIGIT_{d}")


def wording_key(wording_class: str) -> PromptKey | None:
    """``WORDING_<class>`` if such a clip key exists, else None (W_NOREPLY)."""
    try:
        return PromptKey(f"WORDING_{wording_class}")
    except ValueError:
        return None


def copy_key_for(key: PromptKey) -> str:
    """The copy-table key whose text a prompt renders."""
    name = key.value
    if name.startswith("WORDING_"):
        return f"wording.{name.removeprefix('WORDING_')}.spoken"
    return f"prompt.{name.lower()}"


def required_keys(digit_set: tuple[int, ...]) -> frozenset[PromptKey]:
    """Every PromptKey except DIGIT_d for d outside the bank's digit set."""
    out: set[PromptKey] = set()
    for k in PromptKey:
        if k.value.startswith("DIGIT_") and int(k.value.removeprefix("DIGIT_")) not in digit_set:
            continue
        out.add(k)
    return frozenset(out)


def compute_prompts_hash(m: PromptManifest) -> str:
    body = m.model_dump(mode="json", exclude={"prompts_hash"})
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptClip:
    key: PromptKey
    ulaw: bytes
    duration_ms: float
    copy_key: str


class PromptSet:
    """Immutable prompt clips of one language."""

    def __init__(
        self, lang: Lang, prompts_hash: str, voice: str, clips: dict[PromptKey, PromptClip]
    ) -> None:
        self.lang = lang
        self.prompts_hash = prompts_hash
        self.voice = voice
        self._clips = MappingProxyType(dict(clips))

    def get(self, key: PromptKey) -> PromptClip:
        return self._clips[key]  # KeyError if absent

    def has(self, key: PromptKey) -> bool:
        return key in self._clips

    @property
    def placeholder(self) -> bool:
        return self.voice == PLACEHOLDER_VOICE

    def __len__(self) -> int:
        return len(self._clips)


def _integrity(msg: str) -> BankError:
    return BankError(ErrorCode.BANK_INTEGRITY, msg)


def load_prompts(lang: Lang, base: Path, bank: BankManifest, *, prod: bool = False) -> PromptSet:
    """Verify and load ``<base>/<lang>/``. Raises ``BankError`` (BANK_INTEGRITY) on any sha,
    hash, language or missing-key problem; BANK_HASH_REJECTED for a placeholder set in prod."""
    root = Path(base) / lang.value
    try:
        m = PromptManifest.model_validate_json((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise _integrity(f"prompt manifest unreadable: {e}") from e
    if m.lang is not lang:
        raise _integrity(f"prompt manifest lang {m.lang.value} != {lang.value}")
    if compute_prompts_hash(m) != m.prompts_hash:
        raise _integrity("prompts_hash mismatch")
    if prod and m.voice == PLACEHOLDER_VOICE:
        raise BankError(ErrorCode.BANK_HASH_REJECTED, "placeholder prompts refused in prod")
    clips: dict[PromptKey, PromptClip] = {}
    for a in m.assets:
        if a.lang is not lang:
            raise _integrity(f"{a.file}: lang {a.lang.value}")
        try:
            data = (root / a.file).read_bytes()
        except OSError as e:
            raise _integrity(f"{a.file}: {e}") from e
        if hashlib.sha256(data).hexdigest() != a.sha256:
            raise _integrity(f"sha256 mismatch: {a.file}")
        clips[a.key] = PromptClip(a.key, data, float(a.duration_ms), a.copy_key)
    missing = required_keys(bank.digit_set) - clips.keys()
    if missing:
        raise _integrity(f"missing prompt keys: {sorted(k.value for k in missing)}")
    return PromptSet(lang, m.prompts_hash, m.voice, clips)


def silence(duration_ms: float) -> bytes:
    return bytes([MULAW_SILENCE_BYTE]) * max(1, round(duration_ms * BYTES_PER_MS))


def placeholder_prompts(
    lang: Lang, digit_set: tuple[int, ...], *, duration_ms: float = 400.0
) -> PromptSet:
    """Silent placeholder clips for every required key (dev/test/demo only)."""
    clips = {
        k: PromptClip(k, silence(duration_ms), duration_ms, copy_key_for(k))
        for k in sorted(required_keys(digit_set), key=lambda k: k.value)
    }
    h = hashlib.sha256(f"placeholder:{lang.value}:{duration_ms}".encode()).hexdigest()
    return PromptSet(lang, h, PLACEHOLDER_VOICE, clips)


__all__ = [
    "PLACEHOLDER_VOICE",
    "PromptClip",
    "PromptSet",
    "compute_prompts_hash",
    "copy_key_for",
    "digit_key",
    "load_prompts",
    "mark_for_feedback",
    "mark_for_playlist",
    "mark_for_prompt",
    "mark_for_stimulus",
    "placeholder_prompts",
    "required_keys",
    "silence",
    "wording_key",
]
