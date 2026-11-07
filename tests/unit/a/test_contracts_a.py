"""Unit tests for A-lead contracts (constants, rng, languages, stimulus, trial, procedure, result,
evals, copy). Spec IDs are kept in test names."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from sevi.contracts import constants as K
from sevi.contracts.copy import OUTCOME_VALUES, CopyEntry, CopyFile, CopyVariant, OutcomeVar
from sevi.contracts.enums import (
    AnswerSource,
    AnswerStatus,
    BankVariant,
    Channel,
    Condition,
    Lang,
    Mode,
    ModelVersion,
    OffsetSource,
    Owner,
    PresentationKind,
    PromptKey,
    ScoreAction,
    Transducer,
)
from sevi.contracts.evals import GatesFile, ResultFileMeta
from sevi.contracts.languages import DIGIT_SETS, STIMULUS_WORDS, TIMBRE_LANGUAGE
from sevi.contracts.procedure import ProcedureConfig
from sevi.contracts.result import NormCell, NormCellKey, NormsFile, ResultDecision
from sevi.contracts.rng import derive_seed
from sevi.contracts.stimulus import (
    BankManifest,
    CalibrationProvenance,
    DigitAsset,
    TimbreRenderConfig,
    TimingSpec,
    TripletDesign,
)
from sevi.contracts.trial import PresentationSpec, ScoringResult

H = "0" * 64
SID = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b"
MINUS = "\N{MINUS SIGN}"
NOW = datetime(2026, 11, 4, tzinfo=UTC)


# ---------------------------------------------------------------- §4.3.1 worked example
def _next_snr(snr: float, correct: bool, first_wrong_seen: bool) -> tuple[float, bool]:
    """§4.6.1 scored-trial SNR move, written only from the §4.3 constants."""
    first_wrong_seen = first_wrong_seen or not correct
    step = K.STEP_DB if first_wrong_seen else K.STEP_INITIAL_DB
    nxt = snr - step if correct else snr + step
    return min(max(nxt, K.SNR_MIN_DB), K.SNR_MAX_DB), first_wrong_seen


def test_U_ST_010_worked_example_constants() -> None:
    """§4.3.1 table: responses C C C W C W W C from START_SNR_DB."""
    responses = [True, True, True, False, True, False, False, True]
    expected = [  # (presented, first_wrong_seen after, step, next)
        (0, False, 4, -4),
        (-4, False, 4, -8),
        (-8, False, 4, -12),
        (-12, True, 2, -10),
        (-10, True, 2, -12),
        (-12, True, 2, -10),
        (-10, True, 2, -8),
        (-8, True, 2, -10),
    ]
    snr, seen = K.START_SNR_DB, False
    for correct, (presented, seen_after, step, nxt) in zip(responses, expected, strict=True):
        assert snr == presented
        new, seen = _next_snr(snr, correct, seen)
        assert seen is seen_after
        assert abs(new - snr) == step
        assert new == nxt
        snr = new


def test_U_ST_010_silent_caller_trace() -> None:
    """§4.3.1: a silent caller goes 0 -> +2 -> +4 -> +6, then the no-response run hits 3."""
    snr, seen, track, run = K.START_SNR_DB, False, [K.START_SNR_DB], 0
    while run < K.NO_RESPONSE_RUN:
        snr, seen = _next_snr(snr, False, seen)
        track.append(snr)
        run += 1
    assert track == [0.0, 2.0, 4.0, 6.0]


def test_C_CON_A_ceiling_clamp() -> None:
    assert _next_snr(K.SNR_MAX_DB, False, True)[0] == K.SNR_MAX_DB
    assert _next_snr(K.SNR_MIN_DB, True, True)[0] == K.SNR_MIN_DB


# ---------------------------------------------------------------- §4.3 triplet design
def _pool(digit_set: tuple[int, ...]) -> list[tuple[int, int, int]]:
    n = len(digit_set)
    return [
        (digit_set[a], digit_set[(a + d1) % n], digit_set[(a + d2) % n])
        for d1, d2 in K.OFFSET_PAIRS[n]
        for a in range(n)
    ]


@pytest.mark.parametrize(("n", "per_pos_pool", "per_pos_subset"), [(8, 15, 7), (9, 14, 6)])
def test_C_CON_A_balanced_design(n: int, per_pos_pool: int, per_pos_subset: int) -> None:
    digit_set = tuple(range(1, n + 1))
    pool = _pool(digit_set)
    assert len(pool) == K.POOL_SIZE[n] == len(set(pool))
    assert all(len(set(t)) == 3 for t in pool)
    subset = pool[: K.CAL_SUBSET_SIZE[n]]
    for trips, per_pos in ((pool, per_pos_pool), (subset, per_pos_subset)):
        for pos in range(3):
            assert set(Counter(t[pos] for t in trips).values()) == {per_pos}


def test_C_CON_A_procedure_constants() -> None:
    assert K.N_SCORED - K.SRT_FIRST_TRIAL + 2 == 21  # trials 5..24 + virtual 25th
    assert len(K.PRACTICE_SNRS_DB) == 3
    assert K.PRISMA_FRAME_BYTES / 2 / K.SAMPLE_RATE_WIRE_HZ * 1000 == K.PRISMA_FRAME_MS_8K


# ---------------------------------------------------------------- §0 derive_seed
def test_C_CON_A_derive_seed_formula_and_determinism() -> None:
    raw = hashlib.sha256(f"sevi-seed-v1|{SID}|noise:1".encode()).digest()
    assert derive_seed(SID, "noise:1") == int.from_bytes(raw[:8], "big")
    assert derive_seed(SID, "noise:1") == derive_seed(SID, "noise:1")
    assert derive_seed(SID, "noise:1") != derive_seed(SID, "noise:2")
    assert 0 <= derive_seed("", "") < 2**64


def test_C_CON_A_derive_seed_pinned() -> None:
    """Pinned so an accidental change of the domain string or byte order fails loudly."""
    assert derive_seed("bank:test", "noise") == int.from_bytes(
        hashlib.sha256(b"sevi-seed-v1|bank:test|noise").digest()[:8], "big"
    )
    assert derive_seed(SID, "triplets") == 0x6C1BDF871D6B6CF1


# ---------------------------------------------------------------- §4.13 languages
def test_C_CON_A_languages() -> None:
    assert DIGIT_SETS[Lang.TA] == (1, 2, 3, 4, 5, 6, 7, 8)
    assert DIGIT_SETS[Lang.EN] == (1, 2, 3, 4, 5, 6, 8, 9)
    for lang, ds in DIGIT_SETS.items():
        assert tuple(STIMULUS_WORDS[lang]) == ds
        assert len(ds) in K.POOL_SIZE
    assert sum(len(w) for w in STIMULUS_WORDS[Lang.EN].values()) == 31
    assert TIMBRE_LANGUAGE == {Lang.TA: "ta-IN", Lang.EN: "en-IN"}


# ---------------------------------------------------------------- §4.4 stimulus
def _bank(**over: Any) -> dict[str, Any]:
    ds = DIGIT_SETS[Lang.TA]
    d: dict[str, Any] = {
        "lang": "ta",
        "bank_version": "ta-0.0.1",
        "variant": BankVariant.SHIPPED,
        "model_version": ModelVersion.V3,
        "timbre": TimbreRenderConfig(model="timbre-v3.0", voice="TESTTONE", language="ta-IN"),
        "digit_set": ds,
        "digits": tuple(
            DigitAsset(
                digit=x,
                word=STIMULUS_WORDS[Lang.TA][x],
                file=f"digits/{x}.wav",
                sha256=H,
                n_samples=6400,
                rms_dbfs=-23.0,
                offset_db=0.0,
                onset_trim_ms=0.0,
            )
            for x in ds
        ),
        "noise_sha256": H,
        "noise_seed": derive_seed("bank:test", "noise"),
        "design": TripletDesign(
            n_digits=8, offset_pairs=K.OFFSET_PAIRS[8], pool_size=120, calibration_subset_size=56
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
        "created_at": NOW,
        "bank_hash": H,
    }
    d.update(over)
    return d


def test_C_CON_A_bank_manifest_valid_and_roundtrip() -> None:
    m = BankManifest.model_validate(_bank())
    assert m.schema_id == "sevi.bank/1"
    assert BankManifest.model_validate_json(m.model_dump_json()) == m


def test_C_CON_A_bank_manifest_digit_order() -> None:
    b = _bank()
    b["digits"] = tuple(reversed(b["digits"]))
    with pytest.raises(ValidationError, match="digit_set order"):
        BankManifest.model_validate(b)


def test_C_CON_A_bank_manifest_design_size() -> None:
    b = _bank(
        design=TripletDesign(n_digits=9, offset_pairs=(), pool_size=126, calibration_subset_size=54)
    )
    with pytest.raises(ValidationError, match="n_digits"):
        BankManifest.model_validate(b)


def test_C_CON_A_bank_manifest_rejects_non_utc() -> None:
    from datetime import timedelta, timezone

    with pytest.raises(ValidationError):
        BankManifest.model_validate(
            _bank(created_at=datetime(2026, 11, 4, tzinfo=timezone(timedelta(hours=5, minutes=30))))
        )


def test_C_CON_A_timing_spec_is_pinned() -> None:
    assert TimingSpec().lead_ms == K.NOISE_LEAD_MS
    with pytest.raises(ValidationError):
        TimingSpec(lead_ms=400.0)


# ---------------------------------------------------------------- §4.5 trial
def _spec(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "presentation_id": 1,
        "kind": PresentationKind.SCORED,
        "scored_index": 1,
        "snr_db": 0.0,
        "triplet": (4, 7, 2),
        "pool_index": 0,
        "noise_seed": derive_seed(SID, "noise:1"),
    }
    d.update(over)
    return d


def test_C_CON_A_presentation_spec_scored_index() -> None:
    PresentationSpec.model_validate(_spec())
    with pytest.raises(ValidationError, match="scored_index"):
        PresentationSpec.model_validate(_spec(scored_index=None))
    with pytest.raises(ValidationError, match="scored_index"):
        PresentationSpec.model_validate(_spec(kind=PresentationKind.LOUD))


def test_C_CON_A_presentation_spec_feedback_practice_only() -> None:
    PresentationSpec.model_validate(
        _spec(kind=PresentationKind.PRACTICE, scored_index=None, feedback=True)
    )
    with pytest.raises(ValidationError, match="feedback"):
        PresentationSpec.model_validate(_spec(feedback=True))


def test_C_CON_A_triplet_distinct_and_snr_range() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        PresentationSpec.model_validate(_spec(triplet=(4, 4, 2)))
    with pytest.raises(ValidationError):
        PresentationSpec.model_validate(_spec(snr_db=10.5))


def _score(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "presentation_id": 9,
        "mode": Mode.VOICE,
        "status": AnswerStatus.OK,
        "action": ScoreAction.ACCEPT,
        "reask_prompt": None,
        "source": AnswerSource.SPEECH,
        "recognised": (4, 7, 2),
        "triplet_correct": True,
        "position_correct": (True, True, True),
        "valid": True,
        "no_response": False,
        "number_form": False,
    }
    d.update(over)
    return d


def test_C_CON_A_scoring_result_spec_example() -> None:
    """§4.5 ScoringResult example parses from JSON."""
    js = (
        '{"presentation_id":9,"mode":"voice","status":"OK","action":"ACCEPT","reask_prompt":null,'
        '"source":"SPEECH","recognised":[4,7,2],"triplet_correct":true,'
        '"position_correct":[true,true,true],"valid":true,"no_response":false,'
        '"number_form":false,"normalised_tokens":["4","7","2"]}'
    )
    assert ScoringResult.model_validate_json(js).recognised == (4, 7, 2)


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        ({"status": AnswerStatus.PARTIAL}, "triplet_correct requires status OK"),
        ({"action": ScoreAction.REASK}, "reask_prompt set iff"),
        ({"reask_prompt": PromptKey.REASK_THREE}, "reask_prompt set iff"),
        ({"valid": False}, "valid iff"),
        (
            {"status": AnswerStatus.CAPTURE_MISS, "triplet_correct": False, "valid": True},
            "valid iff",
        ),
    ],
)
def test_C_CON_A_scoring_result_consistency(over: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        ScoringResult.model_validate(_score(**over))


# ---------------------------------------------------------------- §4.6 procedure
def test_C_CON_A_procedure_config_condition_requires_skip() -> None:
    design = TripletDesign(
        n_digits=8, offset_pairs=K.OFFSET_PAIRS[8], pool_size=120, calibration_subset_size=56
    )
    base = {"seed_scope": SID, "design": design, "digit_set": DIGIT_SETS[Lang.TA]}
    ProcedureConfig.model_validate(base)
    ProcedureConfig.model_validate(
        {**base, "condition": Condition.N2, "skip_volume_and_practice": True}
    )
    with pytest.raises(ValidationError, match="skip_volume_and_practice"):
        ProcedureConfig.model_validate({**base, "condition": Condition.N2})


# ---------------------------------------------------------------- §4.7 result, §4.10 evals
def _meta() -> ResultFileMeta:
    return ResultFileMeta(
        schema_id="sevi.results.norms/1",
        generated_at=NOW,
        git_sha="a" * 40,
        generator="make results",
        owner=Owner.A,
        contracts_version="0.1.0",
    )


def _cell(lang: Lang) -> NormCell:
    return NormCell(
        key=NormCellKey(
            lang=lang, channel=Channel.BROWSER, mode=Mode.KEYPAD, transducer=Transducer.EARPHONES
        ),
        n=0,
        mean_db=None,
        sd_db=None,
        nh95_db=None,
        cutoff_version=None,
        ages="18-35",
        bank_hash=H,
        computed_at=NOW,
    )


def test_C_CON_A_norms_file_rejects_en() -> None:
    NormsFile(meta=_meta(), cells=(_cell(Lang.TA),))
    with pytest.raises(ValidationError, match="en cells"):
        NormsFile(meta=_meta(), cells=(_cell(Lang.EN),))


def test_C_CON_A_result_file_meta_schema_id() -> None:
    with pytest.raises(ValidationError):
        ResultFileMeta.model_validate({**_meta().model_dump(), "schema_id": "kj.results.x/1"})


def test_C_CON_A_result_decision_spec_example() -> None:
    js = (
        '{"rule":"T13","band":"BORDERLINE","refer_reason":null,"no_band_reason":null,'
        '"wording_class":"W_BORDER","surfaces":["SPOKEN","PAGE","REPORT","OPERATOR"],'
        '"norm_cell":{"lang":"ta","channel":"browser","mode":"keypad","transducer":"earphones"},'
        '"cutoff_version":"v0","nh95_db":-6.1,"n_norm":7,"srt_db":-5.43,'
        '"srt_display":"' + MINUS + '5.5"}'
    )
    d = ResultDecision.model_validate_json(js)
    assert d.srt_display == MINUS + "5.5"
    with pytest.raises(ValidationError):
        ResultDecision.model_validate({**d.model_dump(), "srt_display": "-5.5"})  # ASCII minus
    with pytest.raises(ValidationError):
        ResultDecision.model_validate({**d.model_dump(), "srt_display": MINUS + "5.4"})


def test_C_CON_A_gates_defaults() -> None:
    g = GatesFile.model_validate(
        {
            "meta": _meta(), "g1a": "PENDING", "g1b": "PENDING", "g2": "PENDING",
            "g3t": "PENDING", "g4": "PENDING", "g5": "PENDING", "g6_permits": None,
            "g7": "PENDING", "echo_criterion": "PENDING", "discord_q1": "NO_REPLY",
            "discord_q2": "NO_REPLY", "discord_q3": "NO_REPLY", "discord_q4": "NO_REPLY",
            "friends": "ALLOWED", "phone": "BROWSER_ONLY", "evon": "TEMPLATE",
            "version": "v3", "ing": "PENDING", "twilio_record_ok": None,
            "t_commit_max_ms": None,
        }
    )  # fmt: skip
    assert g.lt05 == "PENDING"
    assert g.t_guard_phone_ms is None


# ---------------------------------------------------------------- §4.11 copy
def test_C_CON_A_outcome_values_cover_outcome_vars() -> None:
    assert set(OUTCOME_VALUES) == {v.value for v in OutcomeVar}
    assert {"NO_PANEL", "RMS", "PRISMA_UNUSABLE"} <= set(OUTCOME_VALUES["SHIP"])


def test_C_CON_A_copy_default_variant_last() -> None:
    default = CopyVariant(variant_id="default", text={"en": "x", "ta": "y"})
    no_panel = CopyVariant(
        variant_id="no_panel", when={OutcomeVar.SHIP: "NO_PANEL"}, text={"en": "x", "ta": "y"}
    )
    CopyEntry(key="pitch.jugaad", surface="README", variants=(no_panel, default))
    with pytest.raises(ValidationError, match="must be last"):
        CopyEntry(key="pitch.jugaad", surface="README", variants=(default, no_panel))


def test_C_CON_A_copy_when_value_checked() -> None:
    with pytest.raises(ValidationError, match="not an outcome value"):
        CopyVariant(variant_id="x", when={OutcomeVar.SHIP: "MAYBE"}, text={"en": "x"})


def test_C_CON_A_copy_file_key_pattern() -> None:
    v = CopyVariant(variant_id="default", text={"en": "x"})
    with pytest.raises(ValidationError):
        CopyEntry(key="bad key!", surface="PAGE", variants=(v,))
    f = CopyFile(
        frozen_at=None,
        banned=("accurate",),
        referral_keywords={"en": ("audiologist",)},
        entries=(CopyEntry(key="ui.x", surface="PAGE", variants=(v,)),),
    )
    assert f.schema_id == "sevi.copy/1"


def test_C_CON_A_no_pending_models() -> None:
    """With A's modules present, every forward reference (e.g. SessionRecord) resolves."""
    from sevi.contracts import ALL_MODELS, PENDING_MODELS

    assert PENDING_MODELS == ()
    assert "sevi.contracts.session.SessionRecord" in {
        f"{m.__module__}.{m.__qualname__}" for m in ALL_MODELS
    }
