"""Helpers for scripts/evon_kaggle.sh (issue #73). Standard library only.

- ``render``: copy ops/evon/kaggle into a push directory with the kernel id, llama.cpp tag,
  limits and API key filled in (the committed files keep placeholders; the key never lands in
  git, only in the private kernel source on Kaggle and in var/run).
- ``parse-url``: read a Kaggle session log on stdin and print the live tunnel URL.
- ``write-env``: write var/run/evon.env (sourced by ``make serve``) atomically, mode 0600.
- ``warmup``: send one tiny chat completion (max_tokens 8, thinking off) so the first real
  session does not hit the llama-server cold start; the key is read from the env file, never
  printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

URL_MARK = re.compile(r"SEVI_EVON_URL=(https://[a-z0-9-]+\.trycloudflare\.com)\b")
STOP_MARK = "SEVI_EVON_STOPPED=1"
KERNEL_SLUG = "sevi-evon"


def parse_url(log: str) -> str | None:
    """Latest tunnel URL announced in the log, or None if none yet or the kernel has stopped.

    The kernel prints a new marker when it restarts a dead tunnel, so the last one wins.
    Kaggle log output may be JSON-escaped or carry timestamps; the regex ignores both.
    """
    urls = [(m.start(), m.group(1)) for m in URL_MARK.finditer(log)]
    if not urls:
        return None
    pos, url = urls[-1]
    if log.find(STOP_MARK, pos) != -1:
        return None
    return url


def env_text(url: str, api_key: str | None = None, max_concurrency: int = 2) -> str:
    """Contents of var/run/evon.env for a running Kaggle server at ``url``."""
    if URL_MARK.fullmatch(f"SEVI_EVON_URL={url}") is None:
        raise ValueError(f"not a quick-tunnel URL: {url!r}")
    lines = [
        "SEVI_EVON__ENABLED=true",
        f"SEVI_EVON__BASE_URL={url}/v1",
        f"SEVI_EVON__MAX_CONCURRENCY={max_concurrency}",
    ]
    if api_key:
        if re.fullmatch(r"[A-Za-z0-9._~+/=-]+", api_key) is None:
            raise ValueError("api key has characters unsafe for a sourced env file")
        lines.append(f"SEVI_EVON__API_KEY={api_key}")
    return "\n".join(lines) + "\n"


def write_env(path: Path, text: str) -> None:
    """Atomic (tmp + rename) and private (0600): the file may hold the session key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def render(
    src: Path, dest: Path, user: str, tag: str, key: str, max_hours: float, idle_min: float
) -> str:
    """Fill the kernel placeholders into ``dest``; returns the kernel id ``user/sevi-evon``."""
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
    kernel_id = f"{user}/{KERNEL_SLUG}"
    meta["id"] = kernel_id
    code = (src / meta["code_file"]).read_text(encoding="utf-8")
    for k, v in {
        "__LLAMA_TAG__": tag,
        "__EVON_API_KEY__": key,
        "__MAX_HOURS__": f"{max_hours:g}",
        "__IDLE_MIN__": f"{idle_min:g}",
    }.items():
        if f'"{k}"' not in code:
            raise ValueError(f"placeholder {k} missing from {meta['code_file']}")
        code = code.replace(f'"{k}"', json.dumps(v), 1)
    out = dest / meta["code_file"]
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    return kernel_id


def stop_path(key: str) -> str:
    """Same derivation as ``stop_path`` in ops/evon/kaggle/evon_server.py."""
    return "/sevi-stop/" + hashlib.sha256(f"sevi-stop:{key}".encode()).hexdigest()[:24]


def ntfy_topic(key: str) -> str:
    """Same derivation as ``announce`` in ops/evon/kaggle/evon_server.py."""
    return "sevi-evon-" + hashlib.sha256(f"sevi-topic:{key}".encode()).hexdigest()[:32]


def ntfy_log(body: str) -> str:
    """ntfy.sh ``/json?poll=1`` lines -> the published messages, one per line, in order."""
    out = []
    for line in body.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("event") == "message":
            out.append(str(ev.get("message", "")))
    return "\n".join(out)


def read_tag(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("tag="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"no tag= line in {path}")


def warmup_body(model: str = "evon") -> dict[str, object]:
    """Request body for the post-boot warm-up: one short turn, 8 tokens, no thinking."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def env_key(text: str) -> str | None:
    """SEVI_EVON__API_KEY from evon.env text, or None."""
    for line in text.splitlines():
        if line.startswith("SEVI_EVON__API_KEY="):
            return line.split("=", 1)[1].strip() or None
    return None


def warmup(url: str, key: str | None, timeout_s: float = 120.0) -> int:
    """POST the warm-up completion to ``url``/v1/chat/completions; return elapsed ms."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(  # noqa: S310 (https tunnel URL, validated by the caller)
        f"{url}/v1/chat/completions",
        data=json.dumps(warmup_body()).encode(),
        headers=headers,
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (https tunnel URL)
        json.loads(resp.read())
    return int((time.monotonic() - t0) * 1000)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evon_kaggle.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--src", type=Path, required=True)
    r.add_argument("--dest", type=Path, required=True)
    r.add_argument("--user", required=True)
    r.add_argument("--tag-file", type=Path, required=True)
    r.add_argument("--max-hours", type=float, default=3.0)
    r.add_argument("--idle-min", type=float, default=30.0)
    sub.add_parser("parse-url")
    sub.add_parser("stop-path")
    sub.add_parser("ntfy-topic")
    sub.add_parser("ntfy-log")
    w = sub.add_parser("write-env")
    w.add_argument("--url", required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--max-concurrency", type=int, default=2)
    wu = sub.add_parser("warmup")
    wu.add_argument("--url", required=True)
    wu.add_argument("--env-file", type=Path, required=True)
    wu.add_argument("--timeout", type=float, default=120.0)
    a = p.parse_args(argv)
    if a.cmd == "render":
        key = os.environ.get("EVON_PUSH_KEY", "")
        if not key:
            print("EVON_PUSH_KEY not set", file=sys.stderr)
            return 2
        print(render(a.src, a.dest, a.user, read_tag(a.tag_file), key, a.max_hours, a.idle_min))
    elif a.cmd == "ntfy-topic":
        print(ntfy_topic(os.environ["EVON_STOP_KEY"]))
    elif a.cmd == "ntfy-log":
        print(ntfy_log(sys.stdin.read()))
    elif a.cmd == "stop-path":
        print(stop_path(os.environ["EVON_STOP_KEY"]))
    elif a.cmd == "parse-url":
        url = parse_url(sys.stdin.read())
        if url is None:
            return 1
        print(url)
    elif a.cmd == "warmup":
        if URL_MARK.fullmatch(f"SEVI_EVON_URL={a.url}") is None:
            print(f"not a quick-tunnel URL: {a.url!r}", file=sys.stderr)
            return 2
        text = a.env_file.read_text() if a.env_file.is_file() else ""
        key = env_key(text) or os.environ.get("SEVI_EVON__API_KEY") or None
        try:
            ms = warmup(a.url, key, a.timeout)
        except Exception as e:  # report any failure, never the key
            print(f"warm-up failed: {type(e).__name__}", file=sys.stderr)
            return 1
        print(f"warm-up ok ({ms} ms)")
    else:
        key = os.environ.get("EVON_ENV_KEY") or None
        write_env(a.out, env_text(a.url, key, a.max_concurrency))
    return 0


if __name__ == "__main__":
    sys.exit(main())
