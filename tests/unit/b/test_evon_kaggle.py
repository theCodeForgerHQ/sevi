"""scripts/evon_kaggle.py and scripts/evon_g4.py helpers (issue #73): URL parsing, env writer."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


ek = _load("evon_kaggle")
URL = "https://quiet-fox-tree-abc.trycloudflare.com"


def test_parse_url_from_kernel_log() -> None:
    log = f"[evon +  300s] llama-server healthy\nSEVI_EVON_URL={URL}\n[evon] alive\n"
    assert ek.parse_url(log) == URL


def test_parse_url_none_before_marker() -> None:
    assert ek.parse_url("[evon +   10s] download ...\n") is None
    # a bare trycloudflare URL (cloudflared's own banner) is not the marker
    assert ek.parse_url(f"INF |  {URL}  |\n") is None


def test_parse_url_last_marker_wins_after_tunnel_restart() -> None:
    other = "https://second-one.trycloudflare.com"
    log = f"SEVI_EVON_URL={URL}\ntunnel died\nSEVI_EVON_URL={other}\n"
    assert ek.parse_url(log) == other


def test_parse_url_none_after_stop() -> None:
    assert ek.parse_url(f"SEVI_EVON_URL={URL}\nidle\nSEVI_EVON_STOPPED=1\n") is None


def test_parse_url_json_escaped_log() -> None:
    log = json.dumps([{"stream_name": "stdout", "data": f"SEVI_EVON_URL={URL}\n"}])
    assert ek.parse_url(log) == URL


def test_parse_url_rejects_other_hosts() -> None:
    assert ek.parse_url("SEVI_EVON_URL=https://evil.example.com\n") is None


def test_env_text_with_and_without_key() -> None:
    t = ek.env_text(URL, "k3y_abc", 2)
    assert t.splitlines() == [
        "SEVI_EVON__ENABLED=true",
        f"SEVI_EVON__BASE_URL={URL}/v1",
        "SEVI_EVON__MAX_CONCURRENCY=2",
        "SEVI_EVON__API_KEY=k3y_abc",
    ]
    assert "API_KEY" not in ek.env_text(URL, None)


@pytest.mark.parametrize("bad", ["http://x.trycloudflare.com", "https://a.b.com", URL + "/v1"])
def test_env_text_rejects_non_tunnel_url(bad: str) -> None:
    with pytest.raises(ValueError, match="quick-tunnel"):
        ek.env_text(bad)


def test_env_text_rejects_shell_unsafe_key() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        ek.env_text(URL, "a b$(x)")


def test_write_env_atomic_private(tmp_path: Path) -> None:
    out = tmp_path / "run" / "evon.env"
    ek.write_env(out, ek.env_text(URL, "abc"))
    ek.write_env(out, ek.env_text(URL, "def"))  # overwrite in place
    assert out.read_text().endswith("SEVI_EVON__API_KEY=def\n")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert not list(out.parent.glob("*.tmp"))


def test_env_file_sources_into_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The written file is what `make serve` sources: Settings must accept it."""
    from sevi.app.settings import Settings

    for line in ek.env_text(URL, "abc").splitlines():
        k, v = line.split("=", 1)
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None).evon  # type: ignore[call-arg]
    assert str(s.base_url).rstrip("/") == f"{URL}/v1"
    assert s.max_concurrency == 2


def test_render_fills_placeholders_and_kernel_id(tmp_path: Path) -> None:
    kid = ek.render(
        ROOT / "ops" / "evon" / "kaggle", tmp_path, "someone", "b12183", "sekret", 2, 15
    )
    assert kid == "someone/sevi-evon"
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == kid
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    code = (tmp_path / meta["code_file"]).read_text()
    assert 'LLAMA_TAG = "b12183"' in code
    assert 'INJECTED_KEY = "sekret"' in code
    assert 'float("2")' in code
    assert 'float("15")' in code
    assert "__EVON_API_KEY__" not in code
    compile(code, "evon_server.py", "exec")


def test_committed_kernel_has_placeholders_only() -> None:
    code = (ROOT / "ops" / "evon" / "kaggle" / "evon_server.py").read_text()
    assert 'INJECTED_KEY = "__EVON_API_KEY__"' in code
    for flag in ("--reasoning-budget", "enable_thinking", "--api-key", "-ts", "--jinja"):
        assert flag in code
    assert ek.read_tag(ROOT / "ops" / "evon" / "LLAMA_CPP_TAG").startswith("b")


def test_g4_percentile_and_env_file(tmp_path: Path) -> None:
    g4 = _load("evon_g4")
    assert g4.pct([100.0, 200.0, 300.0, 400.0], 50) == 200.0
    assert g4.pct([float(x) for x in range(1, 21)], 95) == 19.0
    f = tmp_path / "e.env"
    f.write_text("# c\nSEVI_EVON__BASE_URL=https://x/v1\n")
    assert g4.load_env_file(f) == {"SEVI_EVON__BASE_URL": "https://x/v1"}


def test_stop_path_matches_kernel(tmp_path: Path) -> None:
    ek.render(ROOT / "ops" / "evon" / "kaggle", tmp_path, "u", "b1", "k3y", 1, 1)
    spec = importlib.util.spec_from_file_location("evon_server", tmp_path / "evon_server.py")
    assert spec is not None
    assert spec.loader is not None
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    assert srv.stop_path("k3y") == ek.stop_path("k3y")
    assert ek.stop_path("k3y") != ek.stop_path("other")


def test_ntfy_log_extracts_messages_in_order() -> None:
    body = "\n".join(
        [
            json.dumps({"event": "open"}),
            json.dumps({"event": "message", "message": f"SEVI_EVON_URL={URL}"}),
            "garbage",
            json.dumps({"event": "message", "message": "SEVI_EVON_STOPPED=1"}),
        ]
    )
    log = ek.ntfy_log(body)
    assert log.splitlines() == [f"SEVI_EVON_URL={URL}", "SEVI_EVON_STOPPED=1"]
    assert ek.parse_url(log) is None
    assert ek.parse_url(ek.ntfy_log(body.rsplit("\n", 1)[0])) == URL


def test_ntfy_topic_matches_kernel(tmp_path: Path) -> None:
    import hashlib

    ek.render(ROOT / "ops" / "evon" / "kaggle", tmp_path, "u", "b1", "k3y", 1, 1)
    code = (tmp_path / "evon_server.py").read_text()
    spec = importlib.util.spec_from_file_location("evon_server2", tmp_path / "evon_server.py")
    assert spec is not None
    assert spec.loader is not None
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    assert "SEVI_EVON_STOP=1" in code
    want = "sevi-evon-" + hashlib.sha256(b"sevi-topic:k3y").hexdigest()[:32]
    assert ek.ntfy_topic("k3y") == want
    assert srv.topic("k3y") == want
