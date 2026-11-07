"""Offline scoring accuracy of ``core.scoring`` (EXP-4, issue #61). No live calls.

A labelled corpus of realistic Prisma transcript variants is generated deterministically from
the shipped lexicons (``core/lexicon/{ta,en}.yaml``): for seeded triplets of each language's
digit set, every *form* below is rendered and scored with ``LexiconScorer.score`` exactly as the
app scores a first-window voice answer. Each case carries the digits the speaker actually said
(``truth``); a partial answer's truth is the digits spoken.

Forms (``FORMS``): western numerals spaced / concatenated / punctuated, native Tamil words,
colloquial Tamil, Tanglish romanisations, English words, English-in-Tamil-script, code-mixed,
fillers, extra (non-lexicon) words, partial answers, wrong-language words (the other language's
lexicon), grouping/compound numbers, digit-by-digit across split segments.

Metrics per language and per form: ``triplet_exact`` (recognised == truth), digit-extraction
precision / recall (in-order LCS between recognised and truth digits), status counts and a
``confusion[presented][recognised | "miss"]`` table (positional when the lengths match, else
LCS-aligned; unmatched recognised digits are counted under ``"extra"``).

CLI::

    uv run python -m sevi.calibration.scoring_accuracy                # JSON to stdout
    uv run python -m sevi.calibration.scoring_accuracy --out var/scoring_accuracy.json --report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from sevi.contracts.enums import Lang, Mode, PresentationKind, WindowCloseReason
from sevi.contracts.languages import DIGIT_SETS
from sevi.contracts.rng import derive_seed
from sevi.contracts.trial import AnswerEvidence, PresentationSpec, SttSegment
from sevi.core.scoring import LexiconFile, LexiconScorer, _read_file, load_lexicons
from sevi.core.triplets import build_pool

SEED_SCOPE: Final = "scoring-accuracy"
DEFAULT_N: Final = 12  # triplets per language
LEXICON_DIR: Final = Path(__file__).resolve().parents[1] / "core" / "lexicon"


@dataclass(frozen=True)
class Case:
    lang: Lang
    form: str
    triplet: tuple[int, int, int]
    texts: tuple[str, ...]  # Prisma segments, in order
    truth: tuple[int, ...]  # digits the speaker said


# ---------------------------------------------------------------- lexicon forms
@dataclass(frozen=True)
class Words:
    native: dict[int, str]  # first native-script form
    colloquial: dict[int, str]  # second native-script form when present
    latin: dict[int, str]  # first Latin (Tanglish / English) form
    latin_alt: dict[int, str]  # last Latin form (a less common romanisation)
    tamil_script_en: dict[int, str]  # English word in Tamil script, if listed


def _is_latin(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def words_of(lf: LexiconFile) -> Words:
    eits = set(lf.english_in_tamil_script)
    native: dict[int, str] = {}
    coll: dict[int, str] = {}
    latin: dict[int, str] = {}
    latin_alt: dict[int, str] = {}
    tse: dict[int, str] = {}
    for k, forms in lf.digits.items():
        d = int(k)
        nat = [f for f in forms if not _is_latin(f) and f not in eits]
        lat = [f for f in forms if _is_latin(f)]
        en_ts = [f for f in forms if f in eits]
        if nat:
            native[d] = nat[0]
            coll[d] = nat[1] if len(nat) > 1 else nat[0]
        if lat:
            latin[d] = lat[0]
            latin_alt[d] = lat[-1]
        if en_ts:
            tse[d] = en_ts[0]
    return Words(native, coll, latin, latin_alt, tse)


def _join(t: Sequence[int], m: dict[int, str]) -> str:
    return " ".join(m[d] for d in t)


# ---------------------------------------------------------------- corpus
def triplets_for(lang: Lang, n: int) -> list[tuple[int, int, int]]:
    pool = build_pool(DIGIT_SETS[lang])
    gen = np.random.Generator(np.random.PCG64(derive_seed(SEED_SCOPE, lang.value)))
    return [pool[int(i)] for i in gen.permutation(len(pool))[:n]]


EN_TENS: Final = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy",
                  8: "eighty", 9: "ninety"}  # fmt: skip


HINDI: Final = {1: "ek", 2: "do", 3: "teen", 4: "char", 5: "paanch", 6: "chhah", 7: "saat",
                8: "aath", 9: "nau"}  # fmt: skip
# Tamil combining forms (contracts §4.13 compound rule), as listed in ta.yaml
TA_HUNDREDS: Final = {1: "நூற்று", 2: "இருநூற்று", 3: "முந்நூற்று", 4: "நானூற்று", 5: "ஐநூற்று",
                      6: "அறுநூற்று", 7: "எழுநூற்று", 8: "எண்ணூற்று"}  # fmt: skip
TA_TENS: Final = {2: "இருபத்து", 3: "முப்பத்து", 4: "நாற்பத்து", 5: "ஐம்பத்து", 6: "அறுபத்து",
                  7: "எழுபத்து", 8: "எண்பத்து"}  # fmt: skip


def build_corpus(n: int = DEFAULT_N, lexicon_dir: Path = LEXICON_DIR) -> list[Case]:
    ta = words_of(_read_file(lexicon_dir / "ta.yaml"))
    en = words_of(_read_file(lexicon_dir / "en.yaml"))
    cases: list[Case] = []
    for lang in (Lang.TA, Lang.EN):
        other = en if lang is Lang.TA else ta
        for t in triplets_for(lang, n):
            a, b, c = t
            num = f"{a} {b} {c}"
            forms: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = [
                # known gaps (documented, see tests): a homophone of a digit word as an extra
                # word, and Hindi digit words (not in either lexicon)
                ("extra_homophone", (f"I want to say {num}",), t),
                ("wrong_language_hindi", (_join(t, HINDI),), t),
                ("western_spaced", (num,), t),
                ("western_concat", (f"{a}{b}{c}",), t),
                ("punctuation", (f"{a}, {b}, {c}.",), t),
                ("hyphenated", (f"{a}-{b}-{c}",), t),
                ("english_words", (_join(t, en.latin),), t),
                ("english_alt_spelling", (_join(t, en.latin_alt),), t),
                ("filler_en", (f"um the number is {_join(t, en.latin)}",), t),
                ("extra_words", (f"I think it was {num} okay",), t),
                ("partial_two", (f"{a} {b}",), (a, b)),
                ("partial_one_word", (en.latin[a],), (a,)),
                ("split_segments", (en.latin[a], f"{en.latin[b]} {en.latin[c]}"), t),
            ]
            if lang is Lang.TA:
                forms += [
                    ("native_words", (_join(t, ta.native),), t),
                    ("colloquial", (_join(t, ta.colloquial),), t),
                    ("tanglish", (_join(t, ta.latin),), t),
                    ("tanglish_alt", (_join(t, ta.latin_alt),), t),
                    ("filler_ta", (f"சரி நம்பர் {_join(t, ta.colloquial)}",), t),
                    (
                        "code_mixed",
                        (f"{ta.native[a]} {en.latin[b]} {ta.latin[c]}",),
                        t,
                    ),
                    ("native_with_numeral", (f"{ta.native[a]} {b} {ta.native[c]}",), t),
                ]
                if b >= 2 and a in TA_HUNDREDS and b in TA_TENS:
                    word = f"{TA_HUNDREDS[a]} {TA_TENS[b]} {ta.native[c]}"
                    forms.append(("compound", (word,), t))
                if all(d in ta.tamil_script_en for d in t):
                    forms.append(("english_in_tamil_script", (_join(t, ta.tamil_script_en),), t))
                forms.append(("wrong_language_en_words", (_join(t, en.latin),), t))
            else:
                forms.append(("wrong_language_ta_words", (_join(t, other.native),), t))
                forms.append(("wrong_language_tanglish", (_join(t, other.latin),), t))
                if b >= 2:
                    word = EN_TENS[b] + ("" if c == 0 else f" {en.latin[c]}")
                    forms.append(("grouping", (f"{en.latin[a]} {word}",), t))
                if b >= 2:
                    forms.append(
                        ("compound", (f"{en.latin[a]} hundred and {EN_TENS[b]} {en.latin[c]}",), t)
                    )
            cases += [Case(lang, f, t, texts, truth) for f, texts, truth in forms]
    return cases


# ---------------------------------------------------------------- scoring
def score_case(scorer: LexiconScorer, case: Case) -> dict[str, Any]:
    spec = PresentationSpec(
        presentation_id=1,
        kind=PresentationKind.PRACTICE,
        snr_db=0.0,
        triplet=case.triplet,
        pool_index=0,
        noise_seed=0,
    )
    segs = tuple(
        SttSegment(text=t, segment_index=i + 1, received_ms=0.0, after_commit=False)
        for i, t in enumerate(case.texts)
    )
    ev = AnswerEvidence(
        presentation_id=1,
        lang=case.lang,
        mode=Mode.VOICE,
        window_index=1,
        window_open_mono_ms=0.0,
        window_close_mono_ms=4000.0,
        close_reason=WindowCloseReason.SETTLE,
        segments=segs,
        speech_present=True,
        speech_ms=1500,
        early_speech=False,
        echo_mode=False,
        reask_used=False,
        replay_used=False,
    )
    r = scorer.score(spec, ev)
    return {"status": r.status.value, "recognised": tuple(r.recognised)}


def lcs_pairs(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int]]:
    """Index pairs of an in-order longest common subsequence of ``a`` and ``b``."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _add(conf: dict[str, dict[str, int]], k: str, v: str) -> None:
    row = conf.setdefault(k, {})
    row[v] = row.get(v, 0) + 1


def evaluate(cases: Sequence[Case], scorer: LexiconScorer | None = None) -> dict[str, Any]:
    sc = scorer or load_lexicons()
    rows: list[dict[str, Any]] = []
    for c in cases:
        res = score_case(sc, c)
        rows.append({**asdict(c), "lang": c.lang.value, **res})

    def stats(lang: str, form: str | None = None) -> dict[str, Any]:
        rs = [r for r in rows if r["lang"] == lang and (form is None or r["form"] == form)]
        tp = n_pred = n_true = exact = 0
        conf: dict[str, dict[str, int]] = {}
        status: dict[str, int] = {}
        for r in rs:
            truth, rec = tuple(r["truth"]), tuple(r["recognised"])
            pairs = lcs_pairs(truth, rec)
            tp += len(pairs)
            n_pred += len(rec)
            n_true += len(truth)
            exact += rec == truth
            status[r["status"]] = status.get(r["status"], 0) + 1
            if len(rec) == len(truth):
                for d, g in zip(truth, rec, strict=True):
                    _add(conf, str(d), str(g))
            else:
                hit_t = {i for i, _ in pairs}
                hit_r = {j for _, j in pairs}
                for i, d in enumerate(truth):
                    _add(conf, str(d), str(d) if i in hit_t else "miss")
                for j, g in enumerate(rec):
                    if j not in hit_r:
                        _add(conf, "extra", str(g))
        n = len(rs)
        return {
            "n": n,
            "exact": round(exact / n, 4) if n else None,
            "precision": round(tp / n_pred, 4) if n_pred else None,
            "recall": round(tp / n_true, 4) if n_true else None,
            "status": dict(sorted(status.items())),
            "confusion": {k: dict(sorted(v.items())) for k, v in sorted(conf.items())},
        }

    out: dict[str, Any] = {"experiment": "EXP-4", "n_cases": len(rows), "langs": {}}
    for lang in (Lang.TA, Lang.EN):
        forms = sorted({r["form"] for r in rows if r["lang"] == lang.value})
        out["langs"][lang.value] = {
            "overall": stats(lang.value),
            "by_form": {
                f: {k: v for k, v in stats(lang.value, f).items() if k != "confusion"}
                for f in forms
            },
        }
    out["failures"] = [
        {k: r[k] for k in ("lang", "form", "texts", "truth", "recognised", "status")}
        for r in rows
        if tuple(r["recognised"]) != tuple(r["truth"])
    ]
    return out


def report_markdown(doc: dict[str, Any]) -> str:
    lines = [
        "| Lang | Form | n | Exact | Digit precision | Digit recall |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{100 * v:.0f} %"

    for lg, d in doc["langs"].items():
        o = d["overall"]
        lines.append(
            f"| {lg} | **all** | {o['n']} | {pct(o['exact'])} | {pct(o['precision'])} "
            f"| {pct(o['recall'])} |"
        )
        for f, s in d["by_form"].items():
            lines.append(
                f"| {lg} | {f} | {s['n']} | {pct(s['exact'])} | {pct(s['precision'])} "
                f"| {pct(s['recall'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sevi.calibration.scoring_accuracy")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="triplets per language")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON here")
    ap.add_argument("--report", action="store_true", help="print the markdown table")
    a = ap.parse_args(argv)
    doc = evaluate(build_corpus(a.n))
    text = json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    if a.report:
        print(report_markdown(doc), end="")
    elif a.out is None:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
