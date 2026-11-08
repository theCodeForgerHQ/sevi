"""Web client (issue #17): JS unit tests via `node --test`, encoder parity with core.dsp,
copy build, and page structure / accessibility checks."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from sevi.core.dsp import ULAW_ENCODE_TABLE

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web"
STATIC = WEB / "static"
PAGES = ("index.html", "test.html", "result.html", "r.html")
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _load_build_web_copy():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "build_web_copy", ROOT / "scripts" / "build_web_copy.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _node(*args: str, cwd: Path = WEB, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    assert NODE is not None
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [NODE, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


@needs_node
def test_js_unit_tests_pass() -> None:
    tests = sorted(str(p.relative_to(WEB)) for p in (WEB / "tests").glob("*.test.mjs"))
    assert tests, "no JS tests found"
    r = _node("--test", *tests)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


@needs_node
def test_js_ulaw_encoder_matches_core_dsp_for_every_int16() -> None:
    script = (
        "import('./static/js/g711.js').then(({encodeSample}) => {"
        "const out = new Uint8Array(65536);"
        "for (let s = -32768; s < 32768; s++) out[s + 32768] = encodeSample(s);"
        "process.stdout.write(Buffer.from(out).toString('hex'));});"
    )
    r = _node("--input-type=module", "-e", script)
    assert r.returncode == 0, r.stderr
    assert bytes.fromhex(r.stdout) == ULAW_ENCODE_TABLE.tobytes()


# ---------- copy ----------


def test_build_web_copy_from_fixture(tmp_path: Path) -> None:
    main = _load_build_web_copy().main

    src = tmp_path / "copy.yaml"
    src.write_text(
        """schema_id: sevi.copy/1
frozen_at: null
banned: []
referral_keywords: {en: [], ta: []}
entries:
- key: ui.landing.title
  surface: LANDING
  variants:
  - variant_id: default
    text: {en: Sevi, ta: செவி}
- key: ui.landing.pilot_line
  surface: LANDING
  variants:
  - variant_id: phone_ok
    when: {PHONE: OK}
    text: {en: pilot, ta: pilot-ta}
  - variant_id: default
    text: {en: browser, ta: browser-ta}
- key: prompt.welcome
  surface: SPOKEN
  variants:
  - variant_id: default
    text: {en: hi, ta: hi-ta}
""",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert main(["--src", str(src), "--out", str(out)]) == 0
    en = json.loads((out / "copy.en.json").read_text(encoding="utf-8"))
    ta = json.loads((out / "copy.ta.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in en.items() if not k.startswith("ui.digit.")} == {
        "ui.landing.pilot_line": "browser",
        "ui.landing.title": "Sevi",
    }
    assert ta["ui.landing.title"] == "செவி"
    assert en["ui.digit.1"] == "one"
    assert ta["ui.digit.9"] == "ஒன்பது"
    assert main(["--src", str(src), "--out", str(out), "--check"]) == 0
    (out / "copy.en.json").write_text("{}", encoding="utf-8")
    assert main(["--src", str(src), "--out", str(out), "--check"]) == 1


def test_committed_copy_json_is_fresh() -> None:
    src = ROOT / "src" / "sevi" / "copy" / "copy.yaml"
    if not src.exists():
        pytest.skip("copy.yaml not on this branch yet (A's copy PR)")
    main = _load_build_web_copy().main

    assert main(["--check"]) == 0, "run: uv run python scripts/build_web_copy.py"


def _copy_keys() -> set[str]:
    keys: set[str] = set()
    for lang in ("en", "ta"):
        keys |= set(json.loads((STATIC / "copy" / f"copy.{lang}.json").read_text("utf-8")))
    js = (STATIC / "js" / "copy.js").read_text("utf-8")
    keys |= set(re.findall(r'"((?:ui|consent|share|disclaimer)\.[a-z_.]+)":', js))
    return keys


def test_both_languages_have_the_same_keys() -> None:
    en = json.loads((STATIC / "copy" / "copy.en.json").read_text("utf-8"))
    ta = json.loads((STATIC / "copy" / "copy.ta.json").read_text("utf-8"))
    assert set(en) == set(ta)


def test_every_referenced_copy_key_exists() -> None:
    keys = _copy_keys()
    used: set[str] = set()
    for page in PAGES:
        used |= set(
            re.findall(r'data-copy(?:-aria)?="([^"]+)"', (STATIC / page).read_text("utf-8"))
        )
    for js in (STATIC / "js").glob("*.js"):
        used |= set(re.findall(r'\bt\("([a-z_.A-Z]+)"', js.read_text("utf-8")))
        used |= set(re.findall(r'"(ui\.test\.[a-z_]+)"', js.read_text("utf-8")))
    missing = sorted(used - keys)
    assert not missing, missing


# ---------- structure and accessibility ----------


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang: str | None = None
        self.csp = False
        self.viewport = False
        self.title = False
        self.inline_script = False
        self.inline_style = False
        self.ids: list[str] = []
        self.controls: list[dict[str, str | None]] = []
        self.labels_for: set[str] = set()
        self._stack: list[str] = []
        self._in_label = 0
        self._in_script_src = False
        self._button: dict[str, str | None] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if "style" in a:
            self.inline_style = True
        if any(k.startswith("on") for k in a):
            self.inline_script = True
        if a.get("id"):
            self.ids.append(str(a["id"]))
        if tag == "html":
            self.html_lang = a.get("lang")
        elif tag == "meta" and a.get("http-equiv") == "Content-Security-Policy":
            self.csp = "script-src 'self'" in (a.get("content") or "")
        elif tag == "meta" and a.get("name") == "viewport":
            self.viewport = True
        elif tag == "title":
            self.title = True
        elif tag == "script":
            self._in_script_src = bool(a.get("src"))
            if not a.get("src"):
                self.inline_script = True
        elif tag == "style":
            self.inline_style = True
        elif tag == "label":
            self._in_label += 1
            if a.get("for"):
                self.labels_for.add(str(a["for"]))
        if tag in ("button", "input", "select", "textarea"):
            a["_in_label"] = "1" if self._in_label else None
            a["_text"] = ""
            self.controls.append(a)
            if tag == "button":
                self._button = a

    def handle_data(self, data: str) -> None:
        if self._button is not None:
            self._button["_text"] = (self._button["_text"] or "") + data.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "button":
            self._button = None
        if tag == "label":
            self._in_label -= 1


def _parse(page: str) -> tuple[_Page, str]:
    text = (STATIC / page).read_text("utf-8")
    p = _Page()
    p.feed(text)
    return p, text


@pytest.mark.parametrize("page", PAGES)
def test_page_structure(page: str) -> None:
    p, _ = _parse(page)
    assert p.html_lang in ("ta", "en")
    assert p.csp, "CSP meta with script-src 'self'"
    assert p.viewport
    assert p.title
    assert not p.inline_script, "no inline scripts or on* handlers (CSP)"
    assert not p.inline_style, "no inline styles (CSP style-src 'self')"
    assert len(p.ids) == len(set(p.ids)), "duplicate ids"


@pytest.mark.parametrize("page", PAGES)
def test_every_control_has_an_accessible_name(page: str) -> None:
    p, _ = _parse(page)
    for c in p.controls:
        named = (
            c.get("aria-label")
            or c.get("aria-labelledby")
            or c.get("data-copy")
            or c.get("data-copy-aria")
            or c.get("_in_label")
            or (c.get("id") in p.labels_for)
            or c.get("_text")
        )
        assert named, f"{page}: unnamed control {c}"


SECRET_PATTERNS = [
    re.compile(
        r"(?i)(api[_-]?key|secret|password|bearer\s+[A-Za-z0-9_-]{16,})\s*[:=]\s*['\"][^'\"]{8,}"
    ),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAC[0-9a-f]{32}\b"),  # Twilio account SID
    re.compile(r"(?i)infisical"),
    re.compile(r"\bKJ_|kaanjaanch"),
]


def test_no_secrets_or_old_names_in_static() -> None:
    for f in STATIC.rglob("*"):
        if f.is_file() and f.suffix in {".html", ".js", ".css", ".json"}:
            body = f.read_text("utf-8")
            for pat in SECRET_PATTERNS:
                assert not pat.search(body), f"{f.relative_to(ROOT)}: {pat.pattern}"


def test_no_third_party_or_storage_misuse() -> None:
    for f in STATIC.rglob("*.js"):
        body = f.read_text("utf-8")
        assert "localStorage" not in body, f
        assert "document.cookie" not in body, f
        assert "innerHTML" not in body, f"{f}: use textContent"
    for page in PAGES:
        html = (STATIC / page).read_text("utf-8")
        assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', html), page


def test_test_page_has_cant_hear_keypad_and_live_region() -> None:
    _, html = _parse("test.html")
    assert 'id="cant-hear"' in html
    assert 'data-copy="ui.test.cant_hear"' in html
    assert 'id="keypad"' in html
    assert 'aria-live="polite"' in html
    js = (STATIC / "js" / "test.js").read_text("utf-8")
    assert "for (let d = 1; d <= 9; d++)" in js, "keypad digits 1-9, no 0 (R-21)"


def test_tokens_are_the_only_source_of_raw_values() -> None:
    css = (STATIC / "css" / "sevi.css").read_text("utf-8")
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|oklch\(|rgb\(|hsl\(", body), (
        "colours only in tokens.css"
    )
    assert "font-family: var(" in body
    tokens = (STATIC / "css" / "tokens.css").read_text("utf-8")
    assert "prefers-color-scheme: dark" in tokens
    assert "prefers-reduced-motion" in tokens


def test_config_json_shape() -> None:
    cfg = json.loads((STATIC / "config.json").read_text("utf-8"))
    assert set(cfg) == {"api_base", "ingress", "updated_at"}


def test_web_answer_ms_mirrors_capture_t_answer_ms() -> None:
    """The keypad auto-submit (session.js) must fire before the server's T_ANSWER closes."""
    import re

    from sevi.app.settings import Settings

    js = (Path(__file__).resolve().parents[3] / "web/static/js/session.js").read_text("utf-8")
    m = re.search(r"export const ANSWER_MS = (\d+);", js)
    assert m is not None
    default = Settings.model_fields["capture"].default_factory().t_answer_ms  # type: ignore[misc]
    assert int(m.group(1)) == default


def test_static_page_is_revalidated_not_cached() -> None:
    """After a deploy, browsers must re-check the un-versioned ES modules (live finding)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sevi.app.main import WEB_STATIC, _RevalidatingStatic

    app = FastAPI()
    app.mount("/sevi", _RevalidatingStatic(directory=WEB_STATIC, html=True))
    r = TestClient(app).get("/sevi/js/test.js")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert "etag" in r.headers
