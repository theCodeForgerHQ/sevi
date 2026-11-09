# ruff: noqa: PT018, RUF001, E501
"""Copy-table checks (contracts §4.11, §4.11.2; Person A spec §8.4).

Banned list (whole phrase, case-insensitive), every key in en and ta, every slot fillable, and
the public-face rules for a browser-only product (screening, never diagnosis; a pass never says
hearing is fine).
"""

from __future__ import annotations

import re

import pytest

from sevi.contracts.copy import CopyVariant, OutcomeVar
from sevi.core.wording import (
    REDUCED_SCOPE_OUTCOMES,
    WORDING_SLOTS,
    load_copy,
    placeholders,
    select_variant,
)

COPY = load_copy().copy
BY_KEY = {e.key: e for e in COPY.entries}
# slots filled only from results/* (Person A spec §8.4); none exist in the reduced scope
RESULTS_SLOTS = frozenset(
    {"N_pre_panel", "N_PILOT", "DIFF_DB", "X", "Y", "R", "CI", "Z", "d", "p", "DELTA_P50",
     "DELTA_SNR"}
)  # fmt: skip
# placeholders the web client fills at render time (round counter, queue, report expiry);
# allowed only on browser surfaces, never on spoken/prompt/consent text
CLIENT_SLOTS = frozenset({"k", "n", "when", "pos", "min"})
WEB_SURFACES = frozenset({"LANDING", "PAGE", "REPORT", "SHARE_CARD"})
# web-only strings adopted from the browser client (B's web/static/js/copy.js fallback table)
WEB_CLIENT_KEYS = (
    "ui.landing.read_consent", "ui.landing.not_diagnosis", "ui.landing.language",
    "ui.test.round", "ui.test.connecting", "ui.result.title", "ui.result.expiry_note",
    "ui.result.copied", "ui.report.print", "ui.report.correct", "ui.report.yes",
    "ui.report.no", "ui.report.expires", "ui.wait.ahead", "ui.queue.detail",
)  # fmt: skip
REASSURING = re.compile(
    r"hearing is (fine|normal|ok|good|perfect)|normal hearing confirmed|no hearing loss", re.I
)


def _variants() -> list[tuple[str, CopyVariant]]:
    return [(e.key, v) for e in COPY.entries for v in e.variants]


def _banned_hit(phrase: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None


def test_copy_keys_unique_and_every_variant_has_en_and_ta() -> None:
    assert len(BY_KEY) == len(COPY.entries)
    for key, v in _variants():
        assert set(v.text) == {"en", "ta"}, key
        assert v.text["en"].strip() and v.text["ta"].strip(), key


def test_copy_banned_list_whole_phrase() -> None:
    for key, v in _variants():
        for lang, text in v.text.items():
            for phrase in COPY.banned:
                if phrase.lower() == "this call" and v.when.get(OutcomeVar.PHONE) == "OK":
                    continue  # allowed only under PHONE = OK (contracts §4.11.2 (e))
                assert not _banned_hit(phrase, text), (key, v.variant_id, lang, phrase)


def test_copy_banned_matcher_is_whole_phrase() -> None:
    assert _banned_hit("accurate", "Not Accurate.")
    assert not _banned_hit("accurate", "inaccurately")
    assert not _banned_hit("phone hearing screen", "a phone-quality hearing screen")


def test_copy_never_reassures_and_says_screening() -> None:
    for key, v in _variants():
        for text in v.text.values():
            assert not REASSURING.search(text), key
            assert "Kaan Jaanch" not in text, key
    for key in ("consent.public.page", "disclaimer.page", "prompt.disclaimer", "share.card"):
        assert "not a diagnosis" in BY_KEY[key].variants[-1].text["en"], key


def test_copy_prisma_calibrated_only_under_ship_prisma() -> None:
    for key, v in _variants():
        if "prisma calibrated" in v.text["en"].lower():
            assert v.when.get(OutcomeVar.SHIP) == "PRISMA", key


def test_copy_long_form_says_phone_quality() -> None:
    assert "phone-quality audio" in BY_KEY["ui.landing.subtitle"].variants[0].text["en"]
    for key in ("pitch.jugaad", "post.linkedin", "post.x"):
        for v in BY_KEY[key].variants:
            assert "phone-quality hearing screen" in v.text["en"], key


def test_copy_product_name() -> None:
    title = BY_KEY["ui.landing.title"].variants[0].text
    assert title == {"en": "Sevi", "ta": "செவி"}
    assert "காது சோதனை" in BY_KEY["ui.landing.subtitle"].variants[0].text["ta"]


@pytest.mark.parametrize("lang", ["en", "ta"])
def test_copy_slots_declared_and_consistent(lang: str) -> None:
    for key, v in _variants():
        used = placeholders(v.text[lang])
        assert used == placeholders(v.text["en"]), (key, v.variant_id, "en/ta slots differ")
        if key.startswith("wording."):
            assert used <= WORDING_SLOTS and not v.slots, key
        else:
            assert used == set(v.slots), (key, v.variant_id)
            assert set(v.slots) <= RESULTS_SLOTS | CLIENT_SLOTS, (key, v.variant_id)


def test_copy_reduced_scope_selection_needs_no_results_slots() -> None:
    """Every slot can be filled: with no results files, the selected variants need none."""
    for e in COPY.entries:
        v = select_variant(e, REDUCED_SCOPE_OUTCOMES)
        if not e.key.startswith("wording."):
            assert not set(v.slots) & RESULTS_SLOTS, (e.key, v.variant_id)


def test_copy_client_slots_only_on_web_surfaces() -> None:
    for e in COPY.entries:
        for v in e.variants:
            if set(v.slots) & CLIENT_SLOTS:
                assert e.surface.value in WEB_SURFACES and e.key.startswith("ui."), e.key
                assert not set(v.slots) & RESULTS_SLOTS, e.key


def test_copy_web_client_keys_present() -> None:
    """Keys B's browser client renders (web/static/js/copy.js) exist on a web surface."""
    for key in WEB_CLIENT_KEYS:
        assert BY_KEY[key].surface.value in WEB_SURFACES, key
    assert BY_KEY["ui.test.round"].variants[-1].slots == ("k", "n")
    assert BY_KEY["ui.queue.detail"].variants[-1].slots == ("pos", "min")
    assert "not a diagnosis" in BY_KEY["ui.landing.not_diagnosis"].variants[-1].text["en"]


def test_copy_must_contain_holds_for_every_variant() -> None:
    for e in COPY.entries:
        for v in e.variants:
            for lang, phrases in e.must_contain.items():
                for ph in phrases:
                    assert ph in v.text[lang], (e.key, v.variant_id, lang, ph)


def test_copy_consent_names_processors_and_retention() -> None:
    for key in ("consent.public.page", "consent.study.page", "prompt.consent_public",
                "prompt.consent_study"):  # fmt: skip
        en = BY_KEY[key].variants[0].text["en"]
        for word in ("Gnani", "Tailscale", "Cloudflare", "72 hours"):
            assert word in en, (key, word)
        assert "GitHub" not in en  # the page is served by sevi itself, not GitHub Pages
        assert "we do not keep recordings" not in en.lower()
        assert "after 72 hours" in en


def test_copy_prisma_scoring_claims_gated_on_voice() -> None:
    """A claim that spoken answers are scored by Prisma is unconditional only under VOICE=PUBLIC;
    under DEMO_ONLY it names the teammate demo; under OFF it never appears."""
    claim = re.compile(r"spoken answers are scored", re.I)
    for voice in ("PUBLIC", "DEMO_ONLY", "OFF"):
        for ver in ("v3", "v2.5"):
            vals = {**REDUCED_SCOPE_OUTCOMES, "VOICE": voice, "VERSION": ver}
            for e in COPY.entries:
                text = select_variant(e, vals).text["en"]
                if not claim.search(text):
                    continue
                assert voice != "OFF", e.key
                if voice == "DEMO_ONLY":
                    assert "demo" in text.lower(), e.key


def test_copy_evon_template_explain_en_ta_passes_guard() -> None:
    """``evon.template.explain`` (read by app/explain.py on Evon fallback) passes the guard."""
    from sevi.contracts.enums import Lang
    from sevi.core import guard

    entry = BY_KEY["evon.template.explain"]
    assert [v.variant_id for v in entry.variants] == ["default"]
    text = entry.variants[0].text
    for lang in (Lang.EN, Lang.TA):
        verdict = guard.check(text[lang.value], lang, "EXPLAIN_SPOKEN", banned=COPY.banned)
        assert verdict.ok, (lang, verdict.reasons)
    assert "screening" in text["en"] and "audiologist" in text["en"]
    assert "செவித்திறன் நிபுணர" in text["ta"]
