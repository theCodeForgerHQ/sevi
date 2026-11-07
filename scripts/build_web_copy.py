"""Build web/static/copy/copy.<lang>.json from A's copy table (src/sevi/copy/copy.yaml).

The browser client never hard-codes user-facing text. This script keeps, for every entry on a
surface the browser renders (LANDING, PAGE, REPORT, SHARE_CARD), the default variant's text
(the last variant, `when == {}`; contracts §4.11). Outcome-specific variants are selected
server-side and arrive pre-rendered (e.g. `UiResult.page_text`).

Usage:
    uv run python scripts/build_web_copy.py [--src PATH] [--out DIR] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "src" / "sevi" / "copy" / "copy.yaml"
DEFAULT_OUT = ROOT / "web" / "static" / "copy"
WEB_SURFACES = frozenset({"LANDING", "PAGE", "REPORT", "SHARE_CARD"})
LANGS = ("en", "ta")


def build(copy: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {lang: {} for lang in LANGS}
    for entry in copy["entries"]:
        if entry["surface"] not in WEB_SURFACES:
            continue
        default = entry["variants"][-1]
        if default.get("when"):
            raise ValueError(f"{entry['key']}: last variant must be the default (when == {{}})")
        for lang in LANGS:
            text = default["text"].get(lang)
            if text is None:
                raise ValueError(f"{entry['key']}: missing {lang} text")
            out[lang][entry["key"]] = text
    return {lang: dict(sorted(table.items())) for lang, table in out.items()}


def render(table: dict[str, str]) -> str:
    return json.dumps(table, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="fail if the JSON files are stale")
    args = ap.parse_args(argv)
    if not args.src.exists():
        print(f"build_web_copy: {args.src} not found", file=sys.stderr)
        return 2
    tables = build(yaml.safe_load(args.src.read_text(encoding="utf-8")))
    stale = []
    for lang, table in tables.items():
        path = args.out / f"copy.{lang}.json"
        text = render(table)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if stale:
        print("build_web_copy: stale " + ", ".join(map(str, stale)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
