"""Real-browser end-to-end harness: Chromium (Playwright) drives the shipped web client.

The page under test is the real `web/static` client served by the app (`/sevi/`): landing ->
test.html -> consent -> every trial answered -> result view -> family report (`r.html`).

How the harness knows each answer (no test backdoor in the server): the triplet order is a
pure function of the session id (`Staircase.start`: `PCG64(derive_seed(session_id,
"triplets")).permutation(pool)`; the k-th presentation shows `pool[perm[k-1]]`, whatever the
answers were). The page's own `POST /api/v1/sessions` response carries the session id; an
init script wraps `fetch` and asks Python for the plan before the page continues. Each
`ui phase {state: TEST}` message is one presentation. Practice feedback cross-checks the plan.

Answers:
- keypad: the harness clicks the page's keypad buttons when the server opens the window.
- voice: `navigator.mediaDevices.getUserMedia` is overridden to return a clone of a
  `MediaStreamAudioDestinationNode` stream the harness controls. When the window opens, the
  bank's digit recordings (`<bank>/<lang>/digits/<d>.wav`) are played into it with 250 ms
  gaps, so the real capture worklet -> uplink -> Prisma path carries spoken digits.
  (Chrome's `--use-file-for-fake-audio-capture` is fixed at launch and cannot follow trials.)
  If the server switches to keypad (`ui mode keypad`), the harness falls back to clicking and
  those trials do not count as voice-scored.

Offline (no keys): `SEVI_DEMO_MODE=1 make serve` (fake Prisma "hears" the presented digits once
it detects voiced frames), then
    uv run python scripts/browser_e2e.py --mode keypad --bank-dir stimuli/test-bank
    uv run python scripts/browser_e2e.py --mode voice --bank-dir stimuli/test-bank

LIVE (lead only; real Prisma v3 credits, see the PR for the estimate):
    infisical run --env=dev -- env SEVI_BROWSER_VOICE__ENABLED=true make serve
    uv run python scripts/browser_e2e.py --mode voice --lang ta --screenshots var/e2e-shots

One-time browser download: `uv run playwright install chromium`.
Exit status: 0 all assertions passed, 1 an assertion failed, 2 setup error.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from sevi.contracts.rng import derive_seed
from sevi.core.triplets import build_pool

REPO = Path(__file__).resolve().parents[1]
N_SCORED = 24
MIN_VOICE_SCORED = 20
GAP_MS = 250

INIT_JS = Path(__file__).with_name("browser_e2e_init.js")


@dataclass
class Outcome:
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def check(self, cond: bool, msg: str) -> None:
        if not cond:
            self.ok = False
            self.failures.append(msg)


def triplet_plan(session_id: str, digit_set: tuple[int, ...]) -> list[list[int]]:
    """The presentation order the server's staircase will use for this session."""
    pool = build_pool(digit_set)
    perm = np.random.Generator(np.random.PCG64(derive_seed(session_id, "triplets"))).permutation(
        len(pool)
    )
    return [list(pool[int(i)]) for i in perm]


def load_bank(bank_dir: Path, lang: str) -> tuple[tuple[int, ...], dict[str, str]]:
    root = bank_dir / lang
    manifest = json.loads((root / "manifest.json").read_text())
    digit_set = tuple(int(d) for d in manifest["digit_set"])
    digits = {
        str(d): base64.b64encode((root / "digits" / f"{d}.wav").read_bytes()).decode()
        for d in digit_set
    }
    return digit_set, digits


def parse_answers(spec: str) -> int | None:
    if spec == "correct":
        return None
    m = re.fullmatch(r"wrong-after-(\d+)", spec)
    if not m:
        raise argparse.ArgumentTypeError("--answers: correct | wrong-after-N")
    return int(m.group(1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("keypad", "voice"), default="keypad")
    p.add_argument("--lang", choices=("ta", "en"), default="ta")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/sevi/")
    p.add_argument("--answers", default="correct", help="correct | wrong-after-N (scored trials)")
    p.add_argument("--screenshots", type=Path, default=None, help="directory for page shots")
    p.add_argument(
        "--bank-dir",
        type=Path,
        default=REPO / "stimuli",
        help="bank root the server uses (digit set + digit voices); "
        "stimuli/test-bank for demo mode",
    )
    p.add_argument("--earphones", choices=("yes", "no"), default="yes")
    p.add_argument("--gain-db", type=float, default=0.0, help="voice level vs the bank digits")
    p.add_argument("--timeout-s", type=float, default=600.0, help="whole-session limit")
    p.add_argument("--json-out", type=Path, default=None, help="write the summary JSON here")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def _shot(page: Any, d: Path | None, name: str) -> None:
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(d / f"{name}.png"), full_page=True)


def run(args: argparse.Namespace) -> Outcome:
    from playwright.sync_api import Error as PwError
    from playwright.sync_api import sync_playwright

    wrong_after = parse_answers(args.answers)
    digit_set, digits = load_bank(args.bank_dir, args.lang)
    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(base))
    cfg = {
        "mode": args.mode,
        "digits": digits,
        "digitSet": list(digit_set),
        "gapMs": GAP_MS,
        "gainDb": args.gain_db,
        "wrongAfter": wrong_after,
        "verbose": args.verbose,
    }
    out = Outcome()
    plans: dict[str, list[list[int]]] = {}

    def plan(session_id: str) -> list[list[int]]:
        plans[session_id] = triplet_plan(session_id, digit_set)
        return plans[session_id]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        ctx = browser.new_context(viewport={"width": 390, "height": 844})
        ctx.grant_permissions(["microphone"], origin=origin)
        # The page's API base is the server we test (config.json may point at an ingress).
        ctx.route(
            "**/config.json*",
            lambda r: r.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"api_base": origin, "ingress": "harness", "updated_at": None}),
            ),
        )
        ctx.expose_function("__seviPlan", plan)
        ctx.add_init_script(f"window.__SEVI_CFG = {json.dumps(cfg)};\n{INIT_JS.read_text()}")
        page = ctx.new_page()
        if args.verbose:
            page.on("console", lambda m: print(m.text, file=sys.stderr))
        page.on("pageerror", lambda e: out.failures.append(f"page error: {e}"))

        t0 = time.monotonic()
        try:
            page.goto(base + "index.html")
            page.wait_for_selector("#landing:not([hidden])", timeout=20_000)
            page.check(f"input[name=lang][value={args.lang}]", force=True)
            page.check(f"input[name=earphones][value={args.earphones}]", force=True)
            _shot(page, args.screenshots, "1-landing")
            if args.mode == "voice":
                page.click("#try-voice", timeout=10_000)
            else:
                page.click("#start-keypad", timeout=10_000)
            page.wait_for_url("**/test.html", timeout=10_000)
            page.wait_for_function(
                "() => window.__sevi && window.__sevi.presentations > 0", timeout=60_000
            )
            _shot(page, args.screenshots, "2-test")
            page.wait_for_function(
                "() => ['RESULT', 'DONE', 'ERROR', 'INTERRUPTED', 'EXPIRED']"
                ".includes(document.body.dataset.state)",
                timeout=int(args.timeout_s * 1000),
                polling=250,
            )
            t_result = time.monotonic() - t0
            # The explanation (Evon or template) and `closed` may follow; not required.
            with contextlib.suppress(PwError):
                page.wait_for_function(
                    "() => document.body.dataset.state === 'DONE'", timeout=15_000
                )
        except PwError as e:
            out.check(False, f"session did not finish: {str(e).splitlines()[0]}")
            t_result = time.monotonic() - t0
        state = page.evaluate("() => document.body.dataset.state")
        S: dict[str, Any] = page.evaluate("() => window.__sevi")
        _shot(page, args.screenshots, "3-result")

        out.check(state in ("RESULT", "DONE"), f"page ended in {state}, not RESULT/DONE")
        out.check(not S["errors"], f"harness errors: {S['errors']}")
        out.check(S["plan"] is not None, "never saw the session id (plan missing)")
        res = S["result"]
        out.check(res is not None and bool(res.get("band")), "no result band")
        out.check(
            page.locator("#result-view .result-title").is_visible(), "result view not rendered"
        )
        bad_fb = [f for f in S["feedback"] if not f["correct"]]
        # Keypad scoring is exact, so wrong feedback means the plan is off; in voice it can
        # also be a mishearing (reported, not failed).
        out.check(
            args.mode == "voice" or not bad_fb,
            f"practice feedback says wrong on a correct answer: {bad_fb} "
            "(the triplet plan does not match the server)",
        )

        scored = [a for a in S["answers"] if a["kind"] == "SCORED" and a["window"] == 1]
        by_voice = sum(1 for a in scored if a["by"] == "voice")
        if args.mode == "voice":
            out.check(
                by_voice >= MIN_VOICE_SCORED,
                f"only {by_voice}/{len(scored)} scored trials answered by voice "
                f"(need >= {MIN_VOICE_SCORED})",
            )
            out.check(S["mediaOut"] > 0, "no mic media sent on the uplink")
        if wrong_after is None:
            out.check(
                len({a["scored_index"] for a in scored}) == N_SCORED,
                f"{len(scored)} scored trials answered, expected {N_SCORED}",
            )

        report: dict[str, Any] | None = None
        if res and res.get("report_url"):
            token = res["report_url"].split("#", 1)[1]
            rpage = ctx.new_page()
            try:
                rpage.goto(f"{base}r.html#{token}")
                rpage.wait_for_selector(
                    "#report:not([hidden]), #report-expired:not([hidden])", timeout=20_000
                )
                out.check(rpage.locator("#report").is_visible(), "report page did not open")
                out.check(rpage.locator("#trials tr").count() >= 1, "report has no trials")
                report = {
                    "rows": rpage.locator("#trials tr").count(),
                    "text": rpage.locator("#report-text").inner_text(),
                }
                _shot(rpage, args.screenshots, "4-report")
            except PwError as e:
                out.check(False, f"report page: {str(e).splitlines()[0]}")
        else:
            out.check(False, "no report link in the result")

        ui = [e for e in S["events"] if e["kind"] == "ui"]
        first_test = next(
            (
                e["t"]
                for e in ui
                if e["data"].get("type") == "phase" and e["data"].get("state") == "TEST"
            ),
            None,
        )
        res_t = next((e["t"] for e in ui if e["data"].get("type") == "result"), None)
        out.summary = {
            "mode": args.mode,
            "lang": args.lang,
            "answers": args.answers,
            "final_state": state,
            "session_id": S["sessionId"],
            "presentations": S["presentations"],
            "scored_answered": len(scored),
            "scored_by_voice": by_voice,
            "voice_fallback": S["fallback"],
            "media_frames_out": S["mediaOut"],
            "media_clips_in": S["mediaIn"],
            "practice_feedback": S["feedback"],
            "close_code": S["closed"],
            "timings_s": {
                "to_result": round(t_result, 1),
                "first_trial_to_result": (
                    round((res_t - first_test) / 1000, 1) if res_t and first_test else None
                ),
                "per_presentation": (
                    round((res_t - first_test) / 1000 / S["presentations"], 2)
                    if res_t and first_test and S["presentations"]
                    else None
                ),
            },
            "result": res,
            "explanation": S["explanation"],
            "report": report,
        }
        browser.close()
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = run(args)
    except Exception as e:  # setup: missing browser, server down, bad bank dir
        print(f"browser_e2e: setup error: {e}", file=sys.stderr)
        return 2
    out.summary["ok"] = out.ok
    out.summary["failures"] = out.failures
    text = json.dumps(out.summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(("PASS" if out.ok else "FAIL") + f" ({args.mode}, {args.lang})", file=sys.stderr)
    return 0 if out.ok else 1


if __name__ == "__main__":
    sys.exit(main())
