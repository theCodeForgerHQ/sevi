"""The single stub for Person A's pure ports (B spec §17.3; contracts §5.1).

Used by B's tests, and by `SEVI_ENV=demo` startup only for an A factory that has not merged yet;
prod never imports this module. A's real `dsp`, `triplets` and `staircase` are already on main,
so there is no `StubProcedure` here (use `sevi.core.staircase.PROCEDURE`). Delete this file once
A's `bank`, `scoring` and `result` land.

- `ToneStimulusProvider` (`StimulusProvider`): digit d -> a (300 + 100 d) Hz tone, exact
  `TimingMap` (lead 500 ms, digits 400 ms, gaps 200 ms, lag 500 ms), white noise at `snr_db`
  seeded by `spec.noise_seed`. Same spec -> same bytes.
- `StubScorer` (`AnswerScorer`): the C§4.5.1 rule order; digits are ASCII numerals
  (`re.findall("[1-9]", text)`); intents "theriyala" / "don't know" -> DONT_KNOW,
  "marupadiyum" / "repeat" -> REPEAT; `keyterms` = the C§4.13 stimulus words of the language.
- `StubClassifier` (`ResultClassifier`): T1-T10, T10b, T11 (NO_CELL when there are no norms;
  en: EXPERIMENTAL_LANGUAGE), T15 PASS when norms are present; surfaces per C§4.7.1;
  `wording()` -> `WordingText(text=f"{class}.{surface}")`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from sevi.contracts.constants import CAL_SUBSET_SIZE, OFFSET_PAIRS, POOL_SIZE
from sevi.contracts.enums import (
    AnswerSource,
    AnswerStatus,
    Band,
    BankVariant,
    Channel,
    EndReason,
    FriendsVariant,
    Lang,
    Mode,
    ModelVersion,
    NoBandReason,
    OffsetSource,
    ProcedureTerminal,
    PromptKey,
    ReferReason,
    ScoreAction,
    Surface,
    SystemEvent,
    Transducer,
    TruthRule,
    WordingClass,
)
from sevi.contracts.languages import DIGIT_SETS, STIMULUS_WORDS
from sevi.contracts.ports import WordingText
from sevi.contracts.procedure import SrtResult
from sevi.contracts.result import ResultDecision, ResultInput
from sevi.contracts.stimulus import (
    BankManifest,
    CalibrationProvenance,
    DigitAsset,
    DigitSpan,
    RenderedTrial,
    TimbreRenderConfig,
    TimingMap,
    TripletDesign,
)
from sevi.contracts.trial import AnswerEvidence, PresentationSpec, ScoringResult
from sevi.core.dsp import float_to_pcm16, ulaw_encode

# ---------------------------------------------------------------- stimulus

LEAD_MS, DIGIT_MS, GAP_MS, LAG_MS = 500, 400, 200, 500
WIRE_HZ = 8000
_ZERO_SHA = "0" * 64


def _sha(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def tone_manifest(
    lang: Lang = Lang.TA, model_version: ModelVersion = ModelVersion.V3
) -> BankManifest:
    digit_set = DIGIT_SETS[lang]
    n = len(digit_set)
    words = STIMULUS_WORDS[lang]
    fields: dict[str, object] = {
        "lang": lang,
        "bank_version": f"{lang.value}-0.0.0",
        "variant": BankVariant.SHIPPED,
        "model_version": model_version,
        "timbre": TimbreRenderConfig(
            model="timbre-v3.0", voice="stub", language="ta-IN" if lang is Lang.TA else "en-IN"
        ),
        "digit_set": digit_set,
        "digits": tuple(
            DigitAsset(
                digit=d,
                word=words[d],
                file=f"digits/{d}.wav",
                sha256=_ZERO_SHA,
                n_samples=DIGIT_MS * 16,
                rms_dbfs=-20.0,
                offset_db=0.0,
                onset_trim_ms=0.0,
            )
            for d in digit_set
        ),
        "noise_sha256": _ZERO_SHA,
        "noise_seed": 0,
        "design": TripletDesign(
            n_digits=n,
            offset_pairs=OFFSET_PAIRS[n],
            pool_size=POOL_SIZE[n],
            calibration_subset_size=CAL_SUBSET_SIZE[n],
        ),
        "calibration": CalibrationProvenance(
            source=OffsetSource.RMS,
            iterations=0,
            prisma_model=None,
            prisma_format=None,
            keyterms_sha256=None,
            prisma_sample_rate_hz=None,
            listener_fit_sha256=None,
            panel_outcome_sha256=None,
        ),
        "experimental": True,
        "created_at": datetime(2026, 11, 7, tzinfo=UTC),
    }
    draft = BankManifest.model_validate({**fields, "bank_hash": _ZERO_SHA})
    bank_hash = _sha(draft.model_dump(mode="json", exclude={"bank_hash", "frozen", "frozen_at"}))
    return draft.model_copy(update={"bank_hash": bank_hash})


class ToneStimulusProvider:
    def __init__(self, lang: Lang = Lang.TA, model_version: ModelVersion = ModelVersion.V3) -> None:
        self._manifest = tone_manifest(lang, model_version)

    @property
    def manifest(self) -> BankManifest:
        return self._manifest

    def render_trial(self, spec: PresentationSpec) -> RenderedTrial:
        m = self._manifest
        if any(d not in m.digit_set for d in spec.triplet):
            raise ValueError(f"triplet {spec.triplet} not in digit set {m.digit_set}")
        ms_to_n = WIRE_HZ // 1000
        clip_ms = LEAD_MS + 3 * DIGIT_MS + 2 * GAP_MS + LAG_MS
        x: Any = np.zeros(clip_ms * ms_to_n)
        spans = []
        t = np.arange(DIGIT_MS * ms_to_n) / WIRE_HZ
        for pos, d in enumerate(spec.triplet):
            on = LEAD_MS + pos * (DIGIT_MS + GAP_MS)
            x[on * ms_to_n : (on + DIGIT_MS) * ms_to_n] = 0.1 * np.sin(
                2 * np.pi * (300 + 100 * d) * t
            )
            spans.append(
                DigitSpan.model_validate(
                    {"position": pos + 1, "digit": d, "onset_ms": on, "offset_ms": on + DIGIT_MS}
                )
            )
        rng = np.random.Generator(np.random.PCG64(spec.noise_seed))
        noise_rms = 0.1 / np.sqrt(2) * 10 ** (-spec.snr_db / 20)
        x = np.asarray(x + rng.normal(0.0, noise_rms, x.size), dtype=np.float64)
        peak = float(np.max(np.abs(x)))
        limited = peak > 0.7
        if limited:
            x *= 0.7 / peak
        ulaw = ulaw_encode(float_to_pcm16(x))
        return RenderedTrial(
            presentation_id=spec.presentation_id,
            lang=m.lang,
            triplet=spec.triplet,
            snr_db=spec.snr_db,
            condition=spec.condition,
            noise_seed=spec.noise_seed,
            noise_start_sample=0,
            bank_hash=m.bank_hash,
            ulaw=ulaw,
            n_bytes=len(ulaw),
            timing=TimingMap(
                clip_ms=clip_ms,
                noise_off_ms=clip_ms,
                digits=(spans[0], spans[1], spans[2]),
                last_digit_offset_ms=spans[2].offset_ms,
            ),
            level_limited=limited,
        )


# ---------------------------------------------------------------- scorer

_DONT_KNOW = ("theriyala", "don't know", "dont know")
_REPEAT = ("marupadiyum", "repeat")
_CAPTURE_MISS_EVENTS = {SystemEvent.BREAKER_TRIP, SystemEvent.RECONNECT, SystemEvent.MIC_LOST}


def _digits(text: str) -> list[int]:
    return [int(c) for c in re.findall("[1-9]", text)]


class StubScorer:
    def digits_so_far(self, lang: Lang, texts: Sequence[str]) -> int:
        return min(12, len(_digits(" ".join(texts))))

    def intent_so_far(
        self, lang: Lang, texts: Sequence[str]
    ) -> Literal["DONT_KNOW", "REPEAT"] | None:
        joined = " ".join(texts).lower()
        if _digits(joined):
            return None
        if any(w in joined for w in _DONT_KNOW):
            return "DONT_KNOW"
        if any(w in joined for w in _REPEAT):
            return "REPEAT"
        return None

    def keyterms(self, lang: Lang, model_version: ModelVersion) -> tuple[str, ...]:
        words = tuple(STIMULUS_WORDS[lang].values())
        if model_version is ModelVersion.V2_5:
            return tuple(w for w in words if w.isascii() and w.isalpha())[:100]
        return words[:200]

    def score(self, spec: PresentationSpec, evidence: AnswerEvidence) -> ScoringResult:
        if evidence.presentation_id != spec.presentation_id:
            raise ValueError("evidence.presentation_id != spec.presentation_id")
        ev = evidence
        source = AnswerSource.NONE
        recognised: list[int] = []
        status: AnswerStatus
        texts = [s.text for s in ev.segments]
        if ev.dtmf:  # rule 1
            source = AnswerSource.DTMF
            recognised = [int(p.key) for p in ev.dtmf[:3]]
            status = AnswerStatus.OK if len(recognised) == 3 else AnswerStatus.PARTIAL
        elif ev.mode is Mode.KEYPAD:  # rule 2
            status = AnswerStatus.NO_RESPONSE
        elif _CAPTURE_MISS_EVENTS & set(ev.system_events):  # rule 3
            status = AnswerStatus.CAPTURE_MISS
        elif (
            SystemEvent.STT_ERROR in ev.system_events
            and all(not t.strip() for t in texts)
            and ev.speech_present
        ):  # rule 3b
            status = AnswerStatus.CAPTURE_MISS
        elif ev.early_speech and not ev.echo_mode and not ev.reask_used:  # rule 4
            status = AnswerStatus.EARLY_ANSWER
        elif all(not t.strip() for t in texts):  # rule 5
            status = AnswerStatus.CAPTURE_MISS if ev.speech_present else AnswerStatus.NO_RESPONSE
        else:  # rule 6
            source = AnswerSource.SPEECH
            recognised = _digits(" ".join(texts))[:12]
            intent = self.intent_so_far(ev.lang, texts)
            if len(recognised) == 3:
                status = AnswerStatus.OK
            elif len(recognised) > 3:
                status = AnswerStatus.AMBIGUOUS
            elif recognised:
                status = AnswerStatus.PARTIAL
            elif intent == "REPEAT":
                status = AnswerStatus.REPEAT_REQUEST
            elif intent == "DONT_KNOW":
                status = AnswerStatus.DONT_KNOW
            else:
                status = AnswerStatus.NO_DIGITS
        # rule 7
        action, reask = ScoreAction.ACCEPT, None
        if status is AnswerStatus.EARLY_ANSWER:
            action, reask = ScoreAction.REASK, PromptKey.REASK_SAY_AGAIN
        elif status is AnswerStatus.REPEAT_REQUEST:
            if not ev.replay_used:
                action = ScoreAction.REPLAY
            else:
                status = AnswerStatus.NO_DIGITS
        if status in (AnswerStatus.NO_DIGITS, AnswerStatus.AMBIGUOUS) and not ev.reask_used:
            action, reask = ScoreAction.REASK, PromptKey.REASK_THREE
        rec = tuple(recognised)
        correct = status is AnswerStatus.OK and rec == tuple(spec.triplet)  # rule 8
        pos = tuple(i < len(rec) and rec[i] == spec.triplet[i] for i in range(3))
        return ScoringResult(
            presentation_id=spec.presentation_id,
            mode=ev.mode,
            status=status,
            action=action,
            reask_prompt=reask,
            source=source,
            recognised=rec,
            triplet_correct=correct,
            position_correct=(pos[0], pos[1], pos[2]),
            valid=status is not AnswerStatus.CAPTURE_MISS,
            no_response=status is AnswerStatus.NO_RESPONSE,
            number_form=False,
        )


# ---------------------------------------------------------------- classifier

ALL4 = (Surface.SPOKEN, Surface.PAGE, Surface.REPORT, Surface.OPERATOR)
_PHONE3 = (Surface.SPOKEN, Surface.REPORT, Surface.OPERATOR)


def srt_display(srt_db: float) -> str:
    v = round(srt_db * 2) / 2
    sign = "−" if v < 0 else ""  # noqa: RUF001
    return f"{sign}{abs(v):.1f}"


def _d(
    rule: TruthRule,
    band: Band,
    wc: WordingClass,
    surfaces: tuple[Surface, ...],
    srt: SrtResult | None,
    refer: ReferReason | None = None,
    no_band: NoBandReason | None = None,
) -> ResultDecision:
    db = srt.srt_db if srt is not None else None
    return ResultDecision(
        rule=rule,
        band=band,
        refer_reason=refer,
        no_band_reason=no_band,
        wording_class=wc,
        surfaces=surfaces,
        norm_cell=None,
        cutoff_version=None,
        nh95_db=None,
        n_norm=None,
        srt_db=db,
        srt_display=srt_display(db) if db is not None else None,
    )


def stub_t8_decision(srt: SrtResult | None) -> ResultDecision:
    return _d(TruthRule.T8, Band.FAILED, WordingClass.W_SYSFAIL, ALL4, srt)


class StubClassifier:
    def classify(self, inp: ResultInput) -> ResultDecision:
        s = inp.srt
        er = inp.end_reason
        n24 = s is not None and s.n_scored == 24
        term = s.terminal if s is not None else None
        if term is ProcedureTerminal.VOLUME_FAILED:
            return _d(
                TruthRule.T1,
                Band.REFER,
                WordingClass.W_REF_LOUD,
                ALL4,
                s,
                ReferReason.LOUD_NOT_HEARD,
            )
        if term is ProcedureTerminal.PRACTICE_FAILED:
            return _d(
                TruthRule.T2,
                Band.REFER,
                WordingClass.W_REF_PRACTICE,
                ALL4,
                s,
                ReferReason.PRACTICE_FAILED,
            )
        if s is not None and (term is ProcedureTerminal.LOUD_FAILED or s.loud_wrong_total >= 2):
            return _d(
                TruthRule.T3,
                Band.REFER,
                WordingClass.W_REF_LOUD,
                ALL4,
                s,
                ReferReason.NO_RESPONSE_LOUD,
            )
        if er is EndReason.CANT_HEAR:
            return _d(
                TruthRule.T4, Band.REFER, WordingClass.W_REF_LOUD, ALL4, s, ReferReason.CANT_HEAR
            )
        if s is not None and s.ceiling_fail:
            return _d(
                TruthRule.T5, Band.REFER, WordingClass.W_REF_CEILING, ALL4, s, ReferReason.CEILING
            )
        if s is not None and s.flags > 3:
            return _d(TruthRule.T6, Band.INCONCLUSIVE, WordingClass.W_INCONC, ALL4, s)
        incomplete = {EndReason.CALLER_HANGUP, EndReason.OPERATOR_ABORT, EndReason.SESSION_TIMEOUT}
        if (not n24 and er in incomplete) or term is ProcedureTerminal.NO_RESPONSE_END:
            spoken = er is not EndReason.CALLER_HANGUP or term is ProcedureTerminal.NO_RESPONSE_END
            surf = ALL4 if spoken else (Surface.PAGE, Surface.REPORT, Surface.OPERATOR)
            return _d(TruthRule.T7, Band.INCOMPLETE, WordingClass.W_INC, surf, s)
        if not n24 and er in (EndReason.SERVER_ERROR, EndReason.SOCKET_LOST):
            return stub_t8_decision(s)
        if er.value in ("CONSENT_NO_REPLY", "DIAL_FAILED", "QUEUE_ABANDONED"):
            return _d(
                TruthRule.T9,
                Band.NOT_STARTED,
                WordingClass.W_NOREPLY,
                (Surface.OPERATOR, Surface.REPORT),
                s,
            )
        if er is EndReason.CONSENT_DECLINED:
            return _d(
                TruthRule.T10,
                Band.DECLINED,
                WordingClass.W_DECLINED,
                (Surface.SPOKEN, Surface.PAGE),
                s,
            )
        if s is None or not n24 or s.srt_db is None:  # null-safety: T8
            return stub_t8_decision(s)
        surf = _PHONE3 if inp.channel is Channel.PHONE else ALL4
        if s.srt_db >= 0.0:
            return _d(
                TruthRule.T10B,
                Band.REFER,
                WordingClass.W_REF_ABS,
                surf,
                s,
                ReferReason.SRT_ABSOLUTE,
            )
        reason: NoBandReason | None = None
        if inp.experimental_language:
            reason = NoBandReason.EXPERIMENTAL_LANGUAGE
        elif inp.friends is FriendsVariant.DISALLOWED:
            reason = NoBandReason.FRIENDS_DISALLOWED
        elif inp.mode is Mode.MIXED:
            reason = NoBandReason.MIXED_MODE
        elif inp.transducer is Transducer.SPEAKER:
            reason = NoBandReason.SPEAKER_TRANSDUCER
        elif inp.channel is Channel.PHONE or inp.norms is None:
            reason = NoBandReason.NO_CELL
        if reason is not None:
            uncomparable = (NoBandReason.FRIENDS_DISALLOWED, NoBandReason.SPEAKER_TRANSDUCER)
            wc = (
                WordingClass.W_NOBAND_UNCOMPARABLE
                if reason in uncomparable
                else WordingClass.W_NOBAND_LT6
            )
            return _d(TruthRule.T11, Band.NO_BAND, wc, surf, s, no_band=reason)
        return _d(TruthRule.T15, Band.PASS, WordingClass.W_PASS, surf, s)

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> WordingText:
        wc = decision.wording_class.value
        return WordingText(
            wording_class=wc,
            surface=surface,
            lang=lang.value,
            text=f"{wc}.{surface.value}",
            copy_key=f"wording.{wc}.{surface.value.lower()}",
            variant_id="stub",
        )
