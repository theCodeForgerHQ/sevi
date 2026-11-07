"""Issue #51: user-facing text lives only in web/static/copy/copy.<lang>.json.

The browser JS must not hard-code Tamil script or English sentences. Developer-facing text
(console messages, thrown errors) is allowed; everything else goes through `t(key)`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JS_DIR = ROOT / "web" / "static" / "js"
JS_FILES = sorted(JS_DIR.rglob("*.js"))

TAMIL = re.compile(r"[஀-௿]")
# String literals ('..', "..", `..`), tokenised left to right so adjacent literals don't merge.
LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`(?:[^`\\]|\\.)*`')
# Three or more space-separated words: reads as a sentence, not a key or identifier.
PLACEHOLDER = re.compile(r"\$\{[^}]*\}")
WORDS = re.compile(r"[A-Za-z]{2,}(?: +[A-Za-z'\u2019,.!?]+){2,}")


def _sentences(line: str) -> list[str]:
    return [
        m.group(0)
        for m in LITERAL.finditer(line)
        if WORDS.search(PLACEHOLDER.sub("x", m.group(0)[1:-1]))
    ]


# Developer-facing calls whose literal arguments are not user-facing.
DEV_ONLY = re.compile(r"console\.\w+\(|throw new \w*Error\(|new Error\(")
COMMENT = re.compile(r"^\s*(//|\*|/\*)")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    lines = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if COMMENT.match(line):
            continue
        lines.append((n, line.split(" // ", 1)[0]))
    return lines


def test_js_files_found() -> None:
    assert any(p.name == "copy.js" for p in JS_FILES)


def test_no_tamil_script_in_web_js() -> None:
    hits = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in JS_FILES
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if TAMIL.search(line)
    ]
    assert not hits, f"Tamil text hard-coded in JS (move it to copy.yaml): {hits}"


def test_no_english_sentences_in_web_js() -> None:
    hits = []
    for p in JS_FILES:
        for n, line in _code_lines(p):
            if DEV_ONLY.search(line):
                continue
            hits += [f"{p.relative_to(ROOT)}:{n}: {lit}" for lit in _sentences(line)]
    assert not hits, "English sentences hard-coded in JS (use t(key)):\n" + "\n".join(hits)


def test_copy_js_has_no_fallback_table() -> None:
    src = (JS_DIR / "copy.js").read_text(encoding="utf-8")
    assert "FALLBACK" not in src
    assert not re.search(r'"ui\.[a-z_.]+"\s*:', src), "copy.js must not embed copy entries"


def test_scanner_catches_planted_strings() -> None:
    """Guard the guard: the patterns flag what they are meant to flag."""
    assert TAMIL.search('const x = "சரி";')
    assert _sentences('el.textContent = "Your result is ready";')
    assert _sentences("el.title = `Round ${k} of many`;")
    assert not _sentences('const k = "ui.result.title";')
    assert not _sentences('x = a ? t("ui.a") : t("ui.b");')
