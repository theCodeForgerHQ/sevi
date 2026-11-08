# ruff: noqa: PT018, RUF001, E501
"""Unit tests for core/wording.py and scripts/select_copy.py (Person A spec §8.6: U-WD-01..08)."""

from __future__ import annotations

import importlib.util
import json
import re
import unicodedata
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from sevi.contracts.copy import CopyEntry, CopySurface, OutcomeVar
from sevi.contracts.enums import Lang, PromptKey, Surface, TruthRule, WordingClass
from sevi.contracts.evals import CopySelectionFile
from sevi.core.result import classify
from sevi.core.wording import (
    CLASS_SURFACES,
    MISSING_SLOT,
    REDUCED_SCOPE_OUTCOMES,
    WORDING_SLOTS,
    fill,
    load_copy,
    placeholders,
    select_variant,
    wording_key,
)
from sevi.fakes.scenarios.result_inputs import inp, srt

REPO = Path(__file__).resolve().parents[3]
TABLE = load_copy()
COPY = TABLE.copy
BY_KEY = {e.key: e for e in COPY.entries}


def _select_copy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "select_copy", REPO / "scripts" / "select_copy.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# contracts §4.11.1, reduced scope: the human panel is cut, so ui.panel.* and consent.panel.page
# are not shipped.
NON_WORDING_KEYS = {
    *(f"ui.landing.{k}" for k in (
        "title", "subtitle", "start_keypad", "try_voice", "voice_busy", "watch_video", "pilot_line",
        "lang_ta", "lang_en_experimental", "earphones_question", "earphones_yes", "earphones_no")),
    *(f"ui.test.{k}" for k in (
        "earphones_no_note", "tap_to_start_audio", "allow_mic", "mic_denied_keypad",
        "voice_switched_keypad", "speaker_echo_warning", "consent_agree", "consent_decline",
        "phase_instructions", "phase_volume", "phase_practice", "phase_loud", "phase_test", "listen",
        "listening", "keypad_hint", "hear_again", "key_delete", "key_confirm", "repeat_instructions", "cant_hear",
        "cant_hear_confirm", "cant_hear_yes", "cant_hear_no", "feedback_correct", "feedback_wrong",
        "interrupted", "session_expired", "connection_slow", "start_again")),
    "ui.queue.position", "ui.queue.take_keypad", "ui.wait.room", "ui.rate_limited",
    "ui.voice_unavailable", "ui.back_soon", "ui.result.report_link", "ui.result.share",
    "ui.result.explanation_label", "ui.result.not_on_device", "ui.report.title",
    "ui.report.expired", "ui.report.trials_heading", "ui.error.generic",
    "ui.banner.unsupported_browser",
    "consent.public.page", "consent.study.page", "disclaimer.page", "share.card",
    *(f"prompt.{k}" for k in (
        "consent_public", "consent_study", "instructions", "volume_intro", "volume_up",
        "practice_intro", "practice_correct", "practice_wrong", "practice_retry", "test_intro",
        "reask_say_again", "reask_three", "loud_check", "switch_to_keypad", "disclaimer", "goodbye",
        "goodbye_early")),
    "evon.system.explain", "evon.system.report", "evon.template.explain",
    "outcome.ship", "outcome.lp_shift", "outcome.evon", "outcome.friends", "outcome.phone",
    "outcome.version", "outcome.channel_check",
    "pitch.tagline", "pitch.jugaad", "pitch.outcome_clause", "post.linkedin", "post.x",
    *(f"video.{k}" for k in ("hook", "keypad", "voice", "panels", "disclaimer", "reveal", "credit")),
    "illustration.sentence.ta", "illustration.caption",
    # web-client strings adopted from B's browser client (web/static/js/copy.js)
    *(f"ui.landing.{k}" for k in ("read_consent", "not_diagnosis", "language")),
    "ui.test.round", "ui.test.connecting", "ui.result.title", "ui.result.expiry_note",
    "ui.result.copied", "ui.report.print", "ui.report.correct", "ui.report.yes", "ui.report.no",
    "ui.report.expires", "ui.wait.ahead", "ui.queue.detail",
}  # fmt: skip
WORDING_KEYS = {wording_key(wc, s) for wc, surfs in CLASS_SURFACES.items() for s in surfs}
LIMITS_CLASSES = (
    WordingClass.W_PASS,
    WordingClass.W_PASS_NOTE,
    WordingClass.W_BORDER,
    WordingClass.W_NOBAND_LT6,
    WordingClass.W_NOBAND_RANGE,
    WordingClass.W_NOBAND_UNCOMPARABLE,
)
LIMITS_EN = ["does not rule out hearing loss", "each ear separately", "high pitches", "audiologist"]
OWN_LANG_EN = (
    "If these numbers are not in your strongest language, please take the test again in your "
    "own language."
)


def _texts(entry: CopyEntry) -> list[tuple[str, str]]:
    return [(lang, t) for v in entry.variants for lang, t in v.text.items()]


def _has_digit(text: str) -> bool:
    return any(unicodedata.category(ch) == "Nd" for ch in text)


def test_U_WD_01_copy_keys_exact_and_prompt_assets() -> None:
    assert set(BY_KEY) == NON_WORDING_KEYS | WORDING_KEYS
    for e in COPY.entries:
        for v in e.variants:
            assert set(v.text) == {"en", "ta"}, e.key
    # surfaces = union of the §4.7.1 rows that use each class
    assert CLASS_SURFACES[WordingClass.W_NOREPLY] == (Surface.OPERATOR, Surface.REPORT)
    assert CLASS_SURFACES[WordingClass.W_DECLINED] == (Surface.SPOKEN, Surface.PAGE)
    for wc, surfs in CLASS_SURFACES.items():
        if Surface.SPOKEN in surfs:
            assert f"WORDING_{wc.value}" in PromptKey.__members__


def test_U_WD_02_wording_and_evon_have_no_digits_and_spoken_no_slots() -> None:
    for key, e in BY_KEY.items():
        if key.startswith(("wording.", "evon.")):
            for lang, t in _texts(e):
                assert not _has_digit(t), (key, lang)
        if e.surface in (CopySurface.SPOKEN, CopySurface.PROMPT, CopySurface.CONSENT):
            for lang, t in _texts(e):
                assert not placeholders(t), (key, lang)


NUMERAL_WHITELIST = ("72 hours", "18 or older", "72 மணி நேர", "18 வயது")


def test_U_WD_03_prompt_consent_numerals_only_in_whitelist() -> None:
    for key, e in BY_KEY.items():
        if e.surface in (CopySurface.PROMPT, CopySurface.CONSENT):
            for lang, t in _texts(e):
                stripped = t
                for phrase in NUMERAL_WHITELIST:
                    stripped = stripped.replace(phrase, "")
                assert not _has_digit(stripped), (key, lang)


# rule -> (wording class, surfaces, slots the decision can fill)
_SRT_ONLY = frozenset({"SRT"})
_CELL = frozenset({"SRT", "N", "AGES"})
_NONE: frozenset[str] = frozenset()
REACHABLE = {
    TruthRule.T1: (WordingClass.W_REF_LOUD, _NONE),
    TruthRule.T2: (WordingClass.W_REF_PRACTICE, _NONE),
    TruthRule.T3: (WordingClass.W_REF_LOUD, _NONE),
    TruthRule.T4: (WordingClass.W_REF_LOUD, _NONE),
    TruthRule.T5: (WordingClass.W_REF_CEILING, _NONE),
    TruthRule.T6: (WordingClass.W_INCONC, _NONE),
    TruthRule.T7: (WordingClass.W_INC, _NONE),
    TruthRule.T8: (WordingClass.W_SYSFAIL, _NONE),
    TruthRule.T9: (WordingClass.W_NOREPLY, _NONE),
    TruthRule.T10: (WordingClass.W_DECLINED, _NONE),
    TruthRule.T10B: (WordingClass.W_REF_ABS, _SRT_ONLY),
    TruthRule.T12: (WordingClass.W_REF_SRT, _CELL),
    TruthRule.T13: (WordingClass.W_BORDER, _CELL),
    TruthRule.T14: (WordingClass.W_PASS_NOTE, _CELL),
    TruthRule.T15: (WordingClass.W_PASS, _CELL),
}
T11_SLOTS = {
    WordingClass.W_NOBAND_LT6: _SRT_ONLY,  # NO_CELL / EXPERIMENTAL without a reference cell
    WordingClass.W_NOBAND_UNCOMPARABLE: _SRT_ONLY,  # FRIENDS / SPEAKER have no cell
    WordingClass.W_NOBAND_RANGE: _CELL,
}


def test_U_WD_04_reachable_slots_are_filled() -> None:
    pairs = [(wc, slots) for wc, slots in REACHABLE.values()] + list(T11_SLOTS.items())
    for wc, slots in pairs:
        for surf in CLASS_SURFACES[wc]:
            e = BY_KEY[wording_key(wc, surf)]
            for lang, t in _texts(e):
                assert placeholders(t) <= slots, (e.key, lang, placeholders(t))
                assert placeholders(t) <= WORDING_SLOTS


def test_U_WD_04_rendered_wording_never_shows_missing_slot() -> None:
    cases = [
        inp(),
        inp(srt=srt(srt_db=-5.0)),
        inp(srt=srt(srt_db=-3.0)),
        inp(srt=srt(srt_db=0.5)),
        inp(norms=None),
        inp(hhie_total=20),
    ]
    for c in cases:
        d = classify(c)
        for surf in d.surfaces:
            for lang in Lang:
                assert MISSING_SLOT not in TABLE.wording(d, lang, surf).text


def test_U_WD_05_select_copy_rules(tmp_path: Path) -> None:
    sc = _select_copy()
    vals = sc.outcome_values(tmp_path)
    assert vals["SHIP"] == "NO_PANEL" and vals["CARRIER"] == "NOT_RUN"
    assert vals["PHONE"] == "BROWSER_ONLY" and vals["GRID"] == "NOT_RUN"
    assert vals["LP_SHIFT"] == "WEAK"
    (tmp_path / "panel_outcome.json").write_text(json.dumps({"ship": "HUMAN", "rule_trace": []}))
    assert sc.outcome_values(tmp_path)["SHIP"] == "HUMAN"
    (tmp_path / "gates.json").write_text(json.dumps({"g1b": "FAIL", "g3t": "PENDING"}))
    v = sc.outcome_values(tmp_path)
    assert v["SHIP"] == "PRISMA_UNUSABLE" and v["PHONE"] == "BROWSER_ONLY"
    assert v["VOICE"] == "DEMO_ONLY"
    assert sc.outcome_values(tmp_path, voice="OFF")["VOICE"] == "OFF"
    (tmp_path / "gates.json").write_text(json.dumps({"g3t": "PASS"}))
    assert sc.outcome_values(tmp_path)["PHONE"] == "OK"


def test_U_WD_05_select_copy_writes_selection_file(tmp_path: Path) -> None:
    sc = _select_copy()
    out = tmp_path / "copy_selection.json"
    assert sc.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    sel = CopySelectionFile.model_validate_json(out.read_text())
    assert set(sel.selected) == set(BY_KEY)
    assert sel.selected["pitch.jugaad"] == "no_panel"
    assert sel.selected["outcome.channel_check"] == "default"
    assert sel.outcome_values["VOICE"] == "DEMO_ONLY"
    assert sel.selected["post.linkedin"] == "no_panel"
    out2 = tmp_path / "sel_off.json"
    assert sc.main(["--results", str(tmp_path), "--out", str(out2), "--voice", "OFF"]) == 0
    sel2 = CopySelectionFile.model_validate_json(out2.read_text())
    assert sel2.selected["post.linkedin"] == "no_panel_voice_off"
    assert sel2.selected["video.voice"] == "voice_off"
    t = load_copy(selection_path=str(out))
    assert t.variant("pitch.outcome_clause").variant_id == "no_panel"


def test_U_WD_05_default_variant_must_be_last() -> None:
    with pytest.raises(ValidationError):
        CopyEntry.model_validate(
            {
                "key": "x.y",
                "surface": "PAGE",
                "variants": [
                    {"variant_id": "default", "text": {"en": "a", "ta": "b"}},
                    {"variant_id": "ok", "when": {"PHONE": "OK"}, "text": {"en": "a", "ta": "b"}},
                ],
            }
        )


def test_U_WD_05_first_match_order() -> None:
    e = BY_KEY["pitch.outcome_clause"]
    pick = lambda **kw: select_variant(e, {**REDUCED_SCOPE_OUTCOMES, **kw}).variant_id  # noqa: E731
    assert pick(SHIP="NO_PANEL") == "no_panel"
    assert pick(SHIP="RMS", GRID="NOT_RUN") == "rms_grid_not_run"
    assert pick(SHIP="HUMAN", GRID="NOT_RUN") == "human_grid_not_run"
    assert pick(SHIP="PRISMA", GRID="RAN") == "prisma"
    assert pick(SHIP="RMS", GRID="RAN") == "default"


RENDERED = (CopySurface.PROMPT, CopySurface.CONSENT, CopySurface.SPOKEN)


def test_U_WD_06_rendered_text_budget_and_banned_list() -> None:
    for lang in ("ta", "en"):
        total = sum(
            len(select_variant(e, REDUCED_SCOPE_OUTCOMES).text[lang])
            for e in COPY.entries
            if e.surface in RENDERED
        )
        assert total <= 6000, (lang, total)
    banned = {b.lower() for b in COPY.banned}
    for phrase in (
        "validated", "diagnostic accuracy", "sensitivity", "specificity", "phone call",
        "over the phone", "any phone", "feature phone", "call this number", "this call",
        "phone hearing screen", "detects hearing loss", "accurate",
    ):  # fmt: skip
        assert phrase in banned


def _in_order(text: str, phrases: list[str]) -> bool:
    pos = 0
    for p in phrases:
        i = text.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


@pytest.mark.parametrize("wc", LIMITS_CLASSES, ids=lambda w: w.value)
def test_U_WD_07_limits_sentences_in_order(wc: WordingClass) -> None:
    for surf in ("spoken", "page", "report"):
        e = BY_KEY[f"wording.{wc.value}.{surf}"]
        assert list(e.must_contain["en"]) == LIMITS_EN
        for v in e.variants:
            for lang in ("en", "ta"):
                assert _in_order(v.text[lang], list(e.must_contain[lang])), (e.key, lang)
            first = (
                "A pass" if wc in (WordingClass.W_PASS, WordingClass.W_PASS_NOTE) else "This result"
            )
            assert f"{first} does not rule out hearing loss." in v.text["en"]
            assert (
                "This test plays the sound to both ears together, so it cannot check each ear "
                "separately or high pitches." in v.text["en"]
            )
            assert "If in doubt, see an audiologist." in v.text["en"]


def test_U_WD_07_ref_abs_own_language_sentence() -> None:
    for surf in CLASS_SURFACES[WordingClass.W_REF_ABS]:
        e = BY_KEY[wording_key(WordingClass.W_REF_ABS, surf)]
        ta_sentence = e.must_contain["ta"][-1]
        for v in e.variants:
            assert OWN_LANG_EN in v.text["en"]
            assert ta_sentence in v.text["ta"]


def test_U_WD_08_outcome_variants_cover_every_value() -> None:
    for ship in ("PRISMA", "HUMAN", "RMS", "NO_PANEL", "PRISMA_UNUSABLE"):
        for grid in ("RAN", "NOT_RUN"):
            vals = {**REDUCED_SCOPE_OUTCOMES, "SHIP": ship, "GRID": grid}
            for key in ("pitch.jugaad", "pitch.outcome_clause", "outcome.ship"):
                v = select_variant(BY_KEY[key], vals)
                if grid == "NOT_RUN":
                    assert "DIFF_DB" not in v.slots and "Z" not in v.slots, (key, ship)
                if "Prisma calibrated" in v.text["en"]:
                    assert v.when.get(OutcomeVar.SHIP) == "PRISMA"
    for carrier in ("MATCH", "DIFFERS", "NOT_RUN"):
        select_variant(BY_KEY["outcome.channel_check"], {"CARRIER": carrier})


def test_wording_fills_slots_and_key() -> None:
    d = classify(inp(srt=srt(srt_db=-5.43)))
    w = TABLE.wording(d, Lang.EN, Surface.PAGE)
    assert w.copy_key == "wording.W_BORDER.page" and w.variant_id == "default"
    assert "−5.5 dB SNR" in w.text and w.lang == "en"
    with pytest.raises(KeyError):
        TABLE.wording(
            classify(inp(srt=None, end_reason="CONSENT_DECLINED")), Lang.EN, Surface.REPORT
        )


def test_fill_none_and_unknown() -> None:
    assert fill("{SRT} {N} {X}", {"SRT": "+1.0", "N": None}) == f"+1.0 {MISSING_SLOT} {{X}}"
    assert re.fullmatch(r"[^{}]*", fill("a {SRT}", {"SRT": "0.0"}))
