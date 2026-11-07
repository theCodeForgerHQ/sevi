"""`sevi` CLI (issue #18): arg parsing and operator API calls against the app (TestClient)."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from app_helpers import OP_PASS, OP_USER, running, settings
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from sevi import cli
from sevi.app.main import build_operator_app
from sevi.app.state import AppState


def _client(st: AppState, *, user: str = OP_USER, pw: str = OP_PASS) -> Callable[[], Any]:
    def make() -> httpx.Client:
        c = TestClient(build_operator_app(st), client=("127.0.0.1", 1))
        c.auth = (user, pw)
        c.headers[cli.CSRF_HEADER] = "1"
        return c

    return make


def run(argv: list[str], st: AppState, **kw: Any) -> tuple[int, Any]:
    out = io.StringIO()
    code = cli.main(argv, client=_client(st, **kw), out=out)
    text = out.getvalue()
    return code, json.loads(text) if text.strip() else None


# ---------- parsing ----------


def test_parser_subcommands() -> None:
    with pytest.raises(SystemExit):  # `gate` is reserved for A (results/gates.json)
        cli.build_parser().parse_args(["gate"])
    p = cli.build_parser()
    a = p.parse_args(
        ["ledger", "reconcile", "--key", "A", "--bucket", "DEV", "--dashboard-used", "3"]
    )
    assert (a.cmd, a.action, a.key, a.dashboard_used) == ("ledger", "reconcile", "A", 3.0)
    a = p.parse_args(["tail", "S1", "-f"])
    assert a.session_id == "S1"
    assert a.follow
    assert p.parse_args(["ready"]).min == "ok"
    assert p.parse_args(["station", "--mode", "voice"]).mode == "voice"
    for bad in (["switch", "nope", "on"], ["ledger"], [], ["ready", "--min", "down"]):
        with pytest.raises(SystemExit):
            p.parse_args(bad)


@pytest.mark.parametrize(
    ("name", "raw", "want"),
    [("evon", "on", True), ("prisma", "OFF", False), ("friends", "allowed", "ALLOWED")],
)
def test_switch_values(name: str, raw: str, want: Any) -> None:
    assert cli.parse_switch_value(name, raw) == want


def test_switch_value_rejected() -> None:
    with pytest.raises(cli.CliError):
        cli.parse_switch_value("evon", "maybe")


def test_missing_credentials() -> None:
    with pytest.raises(cli.CliError, match="SEVI_OP_USER"):
        cli.op_client({})
    c = cli.op_client({"SEVI_OP_USER": "u", "SEVI_OP_PASS": "p", "SEVI_OP_URL": "http://x:1"})
    assert c.headers[cli.CSRF_HEADER] == "1"
    assert str(c.base_url) == "http://x:1"


# ---------- against the operator app ----------


def test_cli_operator_calls(tmp_path: Path) -> None:
    with running(tmp_path) as (_, st, _):
        code, rz = run(["ready", "--min", "degraded"], st)
        assert code == 0
        assert rz["status"] in ("ok", "degraded")
        code, sw = run(["switch", "browser_voice", "off"], st)
        assert code == 0
        assert st.switches.browser_voice is False
        code, led = run(["ledger", "show"], st)
        assert code == 0
        assert "keys" in led
        code, rec = run(
            ["ledger", "reconcile", "--key", "A", "--bucket", "DEV", "--dashboard-used", "1.5"], st
        )
        assert code == 0
        assert rec["inserted"] == 1
        code, stn = run(["station"], st)
        assert code == 0
        assert stn["session_id"]
        code, ps = run(["allowlist", "list"], st)
        assert code == 0
        assert ps == []


def test_cli_bad_credentials_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with running(tmp_path) as (_, st, _):
        code, _ = run(["ledger", "show"], st, pw="wrong")
        assert code == cli.EXIT_REFUSED
        err = capsys.readouterr().err
        assert "HTTP 401" in err
        assert "wrong" not in err


def test_cli_dial_phone_disabled(tmp_path: Path) -> None:
    with running(tmp_path) as (_, st, _):
        code, _ = run(["dial", "P-01"], st)
        assert code == cli.EXIT_REFUSED


def test_cli_unreachable() -> None:
    def make() -> httpx.Client:
        return httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5)

    assert cli.main(["ready"], client=make, out=io.StringIO()) == cli.EXIT_USAGE


# ---------- local commands ----------


def test_tail_filters_session(tmp_path: Path) -> None:
    f = tmp_path / "sevi-2026110700.jsonl"
    rows = [{"session_id": "A", "n": 1}, {"session_id": "B"}, {"session_id": "A", "n": 2}]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n" + '{"session_id": "A"')
    seen: dict[Path, int] = {}
    got = [json.loads(x)["n"] for x in cli.session_lines(tmp_path, "A", seen)]
    assert got == [1, 2]
    with f.open("a") as fh:
        fh.write(', "n": 3}\n')
    assert [json.loads(x)["n"] for x in cli.session_lines(tmp_path, "A", seen)] == [3]
    out = io.StringIO()
    assert cli.main(["tail", "A", "--log-dir", str(tmp_path)], out=out) == 0
    assert len(out.getvalue().splitlines()) == 3


def test_allowlist_entry_never_stores_plain_number() -> None:
    fk = Fernet.generate_key().decode()
    e = cli.allowlist_entry(
        participant_id="P-01",
        number="+919800000000",
        kind="TEAMMATE",
        relation="SEVI_TEAM",
        lang="ta",
        expires_at=datetime(2026, 12, 31, 18, 29, 59, tzinfo=UTC),
        fernet_key=fk,
        hmac_key="h" * 32,
    )
    assert "9800000000" not in e.model_dump_json()
    assert Fernet(fk.encode()).decrypt(e.number_ciphertext.encode()) == b"+919800000000"
    with pytest.raises(cli.CliError):
        cli.allowlist_entry(**{**_kw(fk), "number": "98000"})


def _kw(fk: str) -> dict[str, Any]:
    return {
        "participant_id": "P-01",
        "number": "+919800000000",
        "kind": "TEAMMATE",
        "relation": "SEVI_TEAM",
        "lang": "ta",
        "expires_at": datetime(2026, 12, 31, tzinfo=UTC),
        "fernet_key": fk,
        "hmac_key": "h" * 32,
    }


@pytest.mark.asyncio
async def test_local_ledger_summary(tmp_path: Path) -> None:
    s = settings(tmp_path)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    out = await cli.local_ledger_summary(s)
    assert "A" in out["keys"]
    assert out["keys"]["A"]["spent"] == 0
