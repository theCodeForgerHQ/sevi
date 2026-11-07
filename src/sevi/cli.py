"""`sevi` operator CLI (issue #18).

Remote commands talk to the operator listener (`/op/api/*`, default `http://127.0.0.1:8001`,
override with `SEVI_OP_URL`) with HTTP Basic credentials from `SEVI_OP_USER` / `SEVI_OP_PASS`
and the `X-Sevi-Op: 1` CSRF header. Local commands (`serve`, `allowlist add`,
`ledger local-summary`, `tail`) read settings from the environment, like the app.

Exit codes: 0 ok, 1 API refused / gate not passed, 2 usage or connection problem,
78 config invalid (`serve`).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TextIO

import httpx

DEFAULT_OP_URL: Final = "http://127.0.0.1:8001"
CSRF_HEADER: Final = "X-Sevi-Op"
SWITCHES: Final = (
    "evon",
    "prisma",
    "timbre_runtime",
    "phone",
    "browser_voice",
    "demo_mode",
    "friends",
)
BUCKETS: Final = ("CAL", "OPS", "RESERVE", "DEV", "PUBLIC")
EXIT_OK, EXIT_REFUSED, EXIT_USAGE = 0, 1, 2

ClientFactory = Callable[[], httpx.Client]


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


# ---------- operator API client ----------


def op_client(env: dict[str, str] | None = None) -> httpx.Client:
    e = os.environ if env is None else env
    user, pw = e.get("SEVI_OP_USER", ""), e.get("SEVI_OP_PASS", "")
    if not user or not pw:
        raise CliError("set SEVI_OP_USER and SEVI_OP_PASS (operator Basic credentials)")
    return httpx.Client(
        base_url=e.get("SEVI_OP_URL", DEFAULT_OP_URL),
        auth=(user, pw),
        headers={CSRF_HEADER: "1"},
        timeout=15.0,
    )


def _call(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    body: Any = None,
    idempotent: bool = False,
) -> Any:
    headers = {"Idempotency-Key": uuid.uuid4().hex} if idempotent else {}
    try:
        r = client.request(method, path, json=body, headers=headers)
    except httpx.HTTPError as e:
        raise CliError(f"cannot reach operator API: {type(e).__name__}") from None
    if r.status_code >= 400:
        try:
            detail: Any = r.json()
        except ValueError:
            detail = r.text[:500]
        raise CliError(f"HTTP {r.status_code}: {json.dumps(detail)}", EXIT_REFUSED)
    return r.json() if r.content else None


def _print(obj: Any, out: TextIO) -> None:
    out.write(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


# ---------- local helpers ----------


def _settings() -> Any:
    from sevi.app.settings import load_settings

    return load_settings()


def allowlist_entry(
    *,
    participant_id: str,
    number: str,
    kind: str,
    relation: str,
    lang: str,
    expires_at: datetime,
    fernet_key: str,
    hmac_key: str,
    now: datetime | None = None,
) -> Any:
    """Build an `AllowlistEntry`: Fernet-encrypted number + HMAC-SHA256 lookup hash."""
    from cryptography.fernet import Fernet

    from sevi.contracts.enums import AllowlistKind, Lang, RelationLabel
    from sevi.contracts.session import AllowlistEntry

    e164 = number.strip()
    if not (e164.startswith("+") and e164[1:].isdigit() and 8 <= len(e164) <= 16):
        raise CliError("number must be E.164, e.g. +9198xxxxxxxx")
    return AllowlistEntry(
        participant_id=participant_id,
        kind=AllowlistKind(kind),
        number_ciphertext=Fernet(fernet_key.encode()).encrypt(e164.encode()).decode(),
        number_hmac=hmac.new(hmac_key.encode(), e164.encode(), hashlib.sha256).hexdigest(),
        relation_label=RelationLabel(relation),
        default_lang=Lang(lang),
        created_at=now or datetime.now(UTC),
        expires_at=expires_at,
    )


async def _allowlist_put(sqlite_path: Path, entry: Any) -> None:
    from sevi.app.store import SqliteStore

    store = await SqliteStore(sqlite_path).open()
    try:
        await store.allowlist_put(entry)
    finally:
        await store.close()


async def local_ledger_summary(settings: Any) -> dict[str, Any]:
    """Ledger totals straight from the local SQLite file (works with the server down)."""
    from sevi.adapters.ledger import Ledger, TwilioLimits
    from sevi.app.store import SqliteStore
    from sevi.app.telemetry import Telemetry

    s = settings
    store = await SqliteStore(s.sqlite_path).open()
    try:
        ledger = Ledger(
            store,
            s.ledger,
            alias_b=s.key_b_aliased,
            clock=store.clock,
            telemetry=Telemetry(s.log_dir),
            twilio=TwilioLimits(
                trial_minutes_total=s.twilio.trial_minutes_total,
                trial_minutes_stop=s.twilio.trial_minutes_stop,
            ),
        )
        return await ledger.show()
    finally:
        await store.close()


def session_lines(log_dir: Path, session_id: str, files_seen: dict[Path, int]) -> Iterator[str]:
    """New JSONL lines mentioning `session_id` since the last call (offsets in `files_seen`)."""
    for f in sorted(log_dir.glob("sevi-*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            fh.seek(files_seen.get(f, 0))
            while line := fh.readline():
                if not line.endswith("\n"):
                    break  # partial write; re-read next poll
                files_seen[f] = fh.tell()
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("session_id") == session_id:
                    yield line.rstrip("\n")


# ---------- commands ----------


def cmd_serve(_: argparse.Namespace, __: ClientFactory, ___: TextIO) -> int:
    from sevi.app.main import main as app_main

    return app_main()


def cmd_allowlist(a: argparse.Namespace, client: ClientFactory, out: TextIO) -> int:
    if a.action == "list":
        with client() as c:
            _print(_call(c, "GET", "/op/api/participants"), out)
        return EXIT_OK
    s = _settings()
    fk, hk = s.allowlist.fernet_key, s.allowlist.hmac_key
    if fk is None or hk is None:
        raise CliError("set SEVI_ALLOWLIST__FERNET_KEY and SEVI_ALLOWLIST__HMAC_KEY")
    number = os.environ.get("SEVI_ALLOWLIST_NUMBER") or _read_secret("E.164 number: ")
    entry = allowlist_entry(
        participant_id=a.participant_id,
        number=number,
        kind=a.kind,
        relation=a.relation,
        lang=a.lang,
        expires_at=datetime.fromisoformat(a.expires_at),
        fernet_key=fk.get_secret_value(),
        hmac_key=hk.get_secret_value(),
    )
    asyncio.run(_allowlist_put(s.sqlite_path, entry))
    _print({"participant_id": entry.participant_id, "stored": True}, out)  # number never printed
    return EXIT_OK


def _read_secret(prompt: str) -> str:
    import getpass

    return getpass.getpass(prompt)


def cmd_dial(a: argparse.Namespace, client: ClientFactory, out: TextIO) -> int:
    body: dict[str, Any] = {
        "participant_id": a.participant_id,
        "participant_type": a.type,
        "interview": a.interview,
    }
    if a.lang:
        body["lang"] = a.lang
    for k in ("pseudonym", "age_band", "study_form_id"):
        if getattr(a, k):
            body[k] = getattr(a, k)
    with client() as c:
        _print(_call(c, "POST", "/op/api/calls", body=body, idempotent=True), out)
    return EXIT_OK


def parse_switch_value(name: str, raw: str) -> bool | str:
    v = raw.strip().lower()
    if name == "friends" and v in ("allowed", "disallowed"):
        return v.upper()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    raise CliError(
        f"bad value {raw!r} for {name}: use on/off" + ("/allowed/disallowed" * (name == "friends"))
    )


def cmd_switch(a: argparse.Namespace, client: ClientFactory, out: TextIO) -> int:
    body = {"name": a.name, "value": parse_switch_value(a.name, a.value)}
    with client() as c:
        _print(_call(c, "POST", "/op/api/switches", body=body), out)
    return EXIT_OK


def cmd_ledger(a: argparse.Namespace, client: ClientFactory, out: TextIO) -> int:
    if a.action == "local-summary":
        _print(asyncio.run(local_ledger_summary(_settings())), out)
        return EXIT_OK
    with client() as c:
        if a.action == "show":
            _print(_call(c, "GET", "/op/api/ledger"), out)
            return EXIT_OK
        from sevi.adapters.ledger import UNSCHEDULED

        ev = {
            "event_id": uuid.uuid4().hex,
            "key_id": a.key,
            "bucket": a.bucket,
            "purpose": a.purpose,
            "window_id": UNSCHEDULED,
            "origin": "manual",
            "dashboard_used": a.dashboard_used,
            "at": datetime.now(UTC).isoformat(),
        }
        _print(_call(c, "POST", "/op/api/ledger/reconcile", body=ev), out)
    return EXIT_OK


def cmd_tail(a: argparse.Namespace, _: ClientFactory, out: TextIO) -> int:
    log_dir = Path(a.log_dir) if a.log_dir else _settings().log_dir
    if not log_dir.is_dir():
        raise CliError(f"log dir not found: {log_dir}")
    seen: dict[Path, int] = {}
    while True:
        for line in session_lines(log_dir, a.session_id, seen):
            out.write(line + "\n")
        out.flush()
        if not a.follow:
            return EXIT_OK
        time.sleep(a.interval)


def cmd_gate(a: argparse.Namespace, client: ClientFactory, out: TextIO) -> int:
    """Readiness gate for scripts: exit 0 iff `/op/api/readyz` is at least `--min`."""
    with client() as c:
        rz = _call(c, "GET", "/op/api/readyz")
    _print(rz, out)
    rank = {"down": 0, "degraded": 1, "ok": 2}
    return EXIT_OK if rank.get(rz.get("status", "down"), 0) >= rank[a.min] else EXIT_REFUSED


def cmd_station(a: argparse.Namespace, client: ClientFactory, out: TextIO) -> int:
    body = {"lang": a.lang, "mode": a.mode}
    with client() as c:
        _print(_call(c, "POST", "/op/api/stations/sessions", body=body, idempotent=True), out)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sevi", description="Sevi operator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the public and operator listeners").set_defaults(fn=cmd_serve)

    al = sub.add_parser("allowlist", help="list (API) or add (local store) participants")
    al_sub = al.add_subparsers(dest="action", required=True)
    al_sub.add_parser("list")
    add = al_sub.add_parser("add", help="number read from a hidden prompt or env")
    add.add_argument("participant_id", help="P-NN")
    add.add_argument("--kind", default="TEAMMATE", choices=["TEAMMATE", "VOLUNTEER"])
    add.add_argument(
        "--relation", default="SEVI_TEAM", choices=["SEVI_TEAM", "SON", "DAUGHTER", "FAMILY_MEMBER"]
    )
    add.add_argument("--lang", default="ta", choices=["ta", "en"])
    add.add_argument("--expires-at", default="2026-12-31T18:29:59+00:00")
    al.set_defaults(fn=cmd_allowlist)

    d = sub.add_parser("dial", help="place a Twilio trial IVR call (phone switch must be on)")
    d.add_argument("participant_id")
    d.add_argument("--type", default="PILOT", choices=["PILOT", "STUDY"])
    d.add_argument("--lang", choices=["ta", "en"])
    d.add_argument("--interview", action="store_true")
    d.add_argument("--pseudonym")
    d.add_argument("--age-band", dest="age_band")
    d.add_argument("--study-form-id", dest="study_form_id")
    d.set_defaults(fn=cmd_dial)

    sw = sub.add_parser("switch", help="flip a kill switch")
    sw.add_argument("name", choices=SWITCHES)
    sw.add_argument("value", help="on|off (friends: allowed|disallowed)")
    sw.set_defaults(fn=cmd_switch)

    lg = sub.add_parser("ledger", help="credit ledger")
    lg_sub = lg.add_subparsers(dest="action", required=True)
    lg_sub.add_parser("show")
    lg_sub.add_parser("local-summary", help="read the local SQLite ledger (server may be down)")
    rc = lg_sub.add_parser("reconcile", help="record a manual dashboard reading")
    rc.add_argument("--key", required=True, choices=["A", "B"])
    rc.add_argument("--bucket", required=True, choices=BUCKETS)
    rc.add_argument("--purpose", default="dev")
    rc.add_argument("--dashboard-used", dest="dashboard_used", required=True, type=float)
    lg.set_defaults(fn=cmd_ledger)

    t = sub.add_parser("tail", help="print a session's telemetry lines from the local JSONL logs")
    t.add_argument("session_id")
    t.add_argument("-f", "--follow", action="store_true")
    t.add_argument("--log-dir")
    t.add_argument("--interval", type=float, default=1.0)
    t.set_defaults(fn=cmd_tail)

    g = sub.add_parser("gate", help="exit 0 iff readiness is at least --min")
    g.add_argument("--min", default="ok", choices=["ok", "degraded"])
    g.set_defaults(fn=cmd_gate)

    st = sub.add_parser("station", help="create a supervised station session")
    st.add_argument("--lang", default="ta", choices=["ta", "en"])
    st.add_argument("--mode", default="keypad", choices=["keypad", "voice"])
    st.set_defaults(fn=cmd_station)
    return p


def main(
    argv: Sequence[str] | None = None,
    *,
    client: ClientFactory = op_client,
    out: TextIO | None = None,
) -> int:
    a = build_parser().parse_args(argv)
    try:
        code: int = a.fn(a, client, out or sys.stdout)
    except CliError as e:
        sys.stderr.write(f"sevi: {e}\n")
        return e.code
    except KeyboardInterrupt:
        return EXIT_OK
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
