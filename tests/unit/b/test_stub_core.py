"""stub_core (B spec §17.3): the stand-ins for A's ports satisfy the contract Protocols."""

from __future__ import annotations

import hashlib
import json

import pytest

from sevi.contracts.enums import (
    AnswerStatus,
    Band,
    Censor,
    Channel,
    EndReason,
    FriendsVariant,
    Lang,
    Mode,
    ModelVersion,
    NoBandReason,
    ParticipantType,
    PresentationKind,
    ProcedureTerminal,
    PromptKey,
    ScoreAction,
    Surface,
    SystemEvent,
    Transducer,
    TruthRule,
    WindowCloseReason,
    WordingClass,
)
from sevi.contracts.ports import AnswerScorer, ResultClassifier, StimulusProvider
from sevi.contracts.procedure import SrtResult
from sevi.contracts.result import ResultInput
from sevi.contracts.trial import AnswerEvidence, DtmfPress, PresentationSpec, SttSegment
from sevi.fakes.stub_core import (
    StubClassifier,
    StubScorer,
    ToneStimulusProvider,
    srt_display,
    tone_manifest,
)

SID = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b"


def _spec(pid: int = 1, triplet: tuple[int, int, int] = (4, 7, 2)) -> PresentationSpec:
    return PresentationSpec(
        presentation_id=pid,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=-4.0,
        triplet=triplet,
        pool_index=0,
        noise_seed=12345,
    )


def _ev(**kw: object) -> AnswerEvidence:
    base: dict[str, object] = {
        "presentation_id": 1,
        "lang": Lang.TA,
        "mode": Mode.VOICE,
        "window_index": 1,
        "window_open_mono_ms": 0.0,
        "window_close_mono_ms": 3000.0,
        "close_reason": WindowCloseReason.SETTLE,
        "speech_present": True,
        "speech_ms": 900,
        "early_speech": False,
        "echo_mode": False,
        "reask_used": False,
        "replay_used": False,
    }
    base.update(kw)
    return AnswerEvidence.model_validate(base)


def _seg(text: str) -> SttSegment:
    return SttSegment(text=text, received_ms=100.0, after_commit=False)


def test_protocols_satisfied() -> None:
    assert isinstance(ToneStimulusProvider(), StimulusProvider)
    assert isinstance(StubScorer(), AnswerScorer)
    assert isinstance(StubClassifier(), ResultClassifier)


def test_tone_manifest_hash_is_canonical() -> None:
    m = tone_manifest(Lang.EN)
    body = m.model_dump(mode="json", exclude={"bank_hash", "frozen", "frozen_at"})
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert m.bank_hash == hashlib.sha256(raw.encode()).hexdigest()
    assert m.digit_set == (1, 2, 3, 4, 5, 6, 8, 9)


def test_tone_provider_deterministic_and_timing() -> None:
    p = ToneStimulusProvider()
    a, b = p.render_trial(_spec()), p.render_trial(_spec())
    assert a.ulaw == b.ulaw
    assert a.n_bytes == len(a.ulaw) == round(a.timing.clip_ms * 8)
    assert [d.onset_ms for d in a.timing.digits] == [500, 1100, 1700]
    assert a.timing.last_digit_offset_ms == 2100
    with pytest.raises(ValueError, match="not in digit set"):
        p.render_trial(_spec(triplet=(9, 1, 2)))  # 9 is not in the ta digit set


@pytest.mark.parametrize(
    ("kw", "status", "action"),
    [
        ({"dtmf": (DtmfPress(key="4", at_ms=1), DtmfPress(key="7", at_ms=2), DtmfPress(key="2", at_ms=3))}, AnswerStatus.OK, ScoreAction.ACCEPT),  # noqa: E501
        ({"mode": Mode.KEYPAD}, AnswerStatus.NO_RESPONSE, ScoreAction.ACCEPT),
        ({"system_events": (SystemEvent.RECONNECT,), "segments": (_seg("472"),)}, AnswerStatus.CAPTURE_MISS, ScoreAction.ACCEPT),  # noqa: E501
        ({"system_events": (SystemEvent.STT_ERROR,), "segments": (_seg(" "),)}, AnswerStatus.CAPTURE_MISS, ScoreAction.ACCEPT),  # noqa: E501
        ({"early_speech": True, "segments": (_seg("472"),)}, AnswerStatus.EARLY_ANSWER, ScoreAction.REASK),  # noqa: E501
        ({"segments": ()}, AnswerStatus.CAPTURE_MISS, ScoreAction.ACCEPT),
        ({"segments": (), "speech_present": False}, AnswerStatus.NO_RESPONSE, ScoreAction.ACCEPT),
        ({"segments": (_seg("4 7"), _seg("2"))}, AnswerStatus.OK, ScoreAction.ACCEPT),
        ({"segments": (_seg("4 7 2 1"),)}, AnswerStatus.AMBIGUOUS, ScoreAction.REASK),
        ({"segments": (_seg("4"),)}, AnswerStatus.PARTIAL, ScoreAction.ACCEPT),
        ({"segments": (_seg("marupadiyum"),)}, AnswerStatus.REPEAT_REQUEST, ScoreAction.REPLAY),
        ({"segments": (_seg("repeat"),), "replay_used": True}, AnswerStatus.NO_DIGITS, ScoreAction.REASK),  # noqa: E501
        ({"segments": (_seg("theriyala"),)}, AnswerStatus.DONT_KNOW, ScoreAction.ACCEPT),
        ({"segments": (_seg("hmm"),), "reask_used": True}, AnswerStatus.NO_DIGITS, ScoreAction.ACCEPT),  # noqa: E501
    ],
)  # fmt: skip
def test_stub_scorer_rule_order(
    kw: dict[str, object], status: AnswerStatus, action: ScoreAction
) -> None:
    r = StubScorer().score(_spec(), _ev(**kw))
    assert (r.status, r.action) == (status, action)
    if status is AnswerStatus.OK:
        assert r.triplet_correct
        assert r.recognised == (4, 7, 2)
    if action is ScoreAction.REASK:
        assert r.reask_prompt in (PromptKey.REASK_SAY_AGAIN, PromptKey.REASK_THREE)


def test_stub_scorer_helpers() -> None:
    s = StubScorer()
    assert s.digits_so_far(Lang.TA, ["4 7", "2 1 1 1 1 1 1 1 1 1 1"]) == 12
    assert s.intent_so_far(Lang.EN, ["I don't know"]) == "DONT_KNOW"
    assert s.intent_so_far(Lang.EN, ["4 don't know"]) is None
    assert s.keyterms(Lang.EN, ModelVersion.V2_5)[0] == "one"
    assert s.keyterms(Lang.TA, ModelVersion.V3)[0] == "ஒன்று"
    with pytest.raises(ValueError, match="presentation_id"):
        s.score(_spec(pid=2), _ev())


def _srt(**kw: object) -> SrtResult:
    base: dict[str, object] = {
        "n_scored": 24,
        "complete": True,
        "srt_db": -8.25,
        "track_db": (0.0,),
        "virtual_25_db": -8.0,
        "reversals": 8,
        "censor": Censor.NONE,
        "flags": 0,
        "valid_at_ceiling": 0,
        "correct_at_ceiling": 0,
        "max_consec_valid_wrong_at_ceiling": 0,
        "ceiling_fail": False,
        "mode": Mode.KEYPAD,
        "terminal": ProcedureTerminal.COMPLETE,
        "loud_wrong_total": 0,
        "volume_failed": False,
        "practice_failed": False,
    }
    base.update(kw)
    return SrtResult.model_validate(base)


def _inp(srt: SrtResult | None, **kw: object) -> ResultInput:
    base: dict[str, object] = {
        "session_id": SID,
        "lang": Lang.TA,
        "channel": Channel.BROWSER,
        "mode": Mode.KEYPAD,
        "mode_requested": Mode.KEYPAD,
        "transducer": Transducer.EARPHONES,
        "participant_type": ParticipantType.PUBLIC,
        "end_reason": EndReason.COMPLETED,
        "srt": srt,
        "friends": FriendsVariant.ALLOWED,
        "experimental_language": False,
        "bank_hash": "0" * 64,
        "norms": None,
    }
    base.update(kw)
    return ResultInput.model_validate(base)


@pytest.mark.parametrize(
    ("srt", "kw", "rule", "band"),
    [
        (_srt(terminal=ProcedureTerminal.VOLUME_FAILED, n_scored=0, complete=False, srt_db=None), {}, TruthRule.T1, Band.REFER),  # noqa: E501
        (_srt(), {"end_reason": EndReason.CANT_HEAR}, TruthRule.T4, Band.REFER),
        (_srt(ceiling_fail=True), {}, TruthRule.T5, Band.REFER),
        (_srt(flags=4), {}, TruthRule.T6, Band.INCONCLUSIVE),
        (_srt(n_scored=10, complete=False, srt_db=None, terminal=None), {"end_reason": EndReason.CALLER_HANGUP}, TruthRule.T7, Band.INCOMPLETE),  # noqa: E501
        (_srt(n_scored=10, complete=False, srt_db=None, terminal=None), {"end_reason": EndReason.SOCKET_LOST}, TruthRule.T8, Band.FAILED),  # noqa: E501
        (None, {"end_reason": EndReason.DIAL_FAILED}, TruthRule.T9, Band.NOT_STARTED),
        (None, {"end_reason": EndReason.CONSENT_DECLINED}, TruthRule.T10, Band.DECLINED),
        (_srt(srt_db=0.5), {"experimental_language": True}, TruthRule.T10B, Band.REFER),
        (_srt(), {}, TruthRule.T11, Band.NO_BAND),
        (_srt(n_scored=12, complete=False, srt_db=None, terminal=None), {}, TruthRule.T8, Band.FAILED),  # noqa: E501
    ],
)  # fmt: skip
def test_stub_classifier_truth_table(
    srt: SrtResult | None, kw: dict[str, object], rule: TruthRule, band: Band
) -> None:
    d = StubClassifier().classify(_inp(srt, **kw))
    assert (d.rule, d.band) == (rule, band)


def test_stub_classifier_t11_reasons_and_wording() -> None:
    c = StubClassifier()
    d = c.classify(_inp(_srt(), lang=Lang.EN, experimental_language=True))
    assert d.no_band_reason is NoBandReason.EXPERIMENTAL_LANGUAGE
    d = c.classify(_inp(_srt(), transducer=Transducer.SPEAKER))
    assert d.wording_class is WordingClass.W_NOBAND_UNCOMPARABLE
    d = c.classify(_inp(_srt(), channel=Channel.PHONE))
    assert d.surfaces == (Surface.SPOKEN, Surface.REPORT, Surface.OPERATOR)
    assert d.srt_display == "\u22128.0"
    w = c.wording(d, Lang.TA, Surface.PAGE)
    assert w.text == "W_NOBAND_LT6.PAGE"
    assert w.copy_key == "wording.W_NOBAND_LT6.page"
    assert srt_display(2.26) == "2.5"
