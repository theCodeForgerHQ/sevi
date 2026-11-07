"""Sevi Evon server on a free Kaggle 2x T4 session (issue #73; contracts §1.4, B spec §5.6.1).

Runs as a Kaggle *script kernel* pushed by ``scripts/evon_kaggle.sh push``. Steps:

1. check the GPUs (2x T4 expected; Q4_K_M needs both, ``-ts 1,1``);
2. install the pinned llama.cpp release (prebuilt CUDA 12.8 binary + cudart; source build
   with ``-DCMAKE_CUDA_ARCHITECTURES=75`` only if the prebuilt binary cannot see the GPUs);
3. download the Q4_K_M GGUF from Hugging Face to local disk (not /kaggle/working);
4. start ``llama-server`` with the contract flags (thinking off, ``--api-key``) on 127.0.0.1;
5. expose it with a cloudflared quick tunnel and print one marker line ``SEVI_EVON_URL=<url>``
   that ``scripts/evon_kaggle.sh status`` reads from the live session log;
6. stay up until the hard cap (``MAX_HOURS``), ``IDLE_MIN`` minutes without a completion, or a
   dead llama-server. A dead tunnel is restarted and a new marker line printed.

The API key comes from the Kaggle secret ``SEVI_EVON_API_KEY`` when it is attached to this
notebook, else from the value the push script injected into the uploaded copy (a fresh random
key per session unless SEVI_EVON__API_KEY was set at push time). The key is never printed.
This is a short-lived personal demo endpoint: it shuts itself down; stop it after use.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# ---- injected by scripts/evon_kaggle.sh push (placeholders in the committed file) ----
LLAMA_TAG = "__LLAMA_TAG__"
INJECTED_KEY = "__EVON_API_KEY__"
MAX_HOURS = float("__MAX_HOURS__")
IDLE_MIN = float("__IDLE_MIN__")
# --------------------------------------------------------------------------------------

HF_REPO = "luminenio/gnani-evon-v3.3-30B-A3B-GGUF"
GGUF = "gnani-evon-v3.3-30B-A3B-Q4_K_M.gguf"
GGUF_FALLBACK = "gnani-evon-v3.3-30B-A3B-IQ4_XS.gguf"  # if local disk < 30 GiB free
CUDA = "12.8"
CLOUDFLARED = "2026.10.7"
PORT = 8080
WORK = Path("/tmp/evon")  # noqa: S108  # local disk; /kaggle/working is capped and is saved as output
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[evon +{time.monotonic() - T0:6.0f}s] {msg}", flush=True)


def sh(cmd: str, check: bool = True) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)  # noqa: S602
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd!r} failed: {r.stderr[-2000:]}")
    return r.stdout


def api_key() -> str:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        key = str(UserSecretsClient().get_secret("SEVI_EVON_API_KEY"))
        if key:
            log("api key: Kaggle secret SEVI_EVON_API_KEY")
            return key
    except Exception:  # noqa: S110  # no secret attached: use the injected key
        pass
    if INJECTED_KEY.startswith("__"):
        sys.exit("no API key: attach Kaggle secret SEVI_EVON_API_KEY or push via the script")
    log("api key: injected at push time")
    return INJECTED_KEY


def gpus() -> list[str]:
    out = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
    names = [line.strip() for line in out.splitlines() if line.strip()]
    log(f"gpus: {names}")
    if len(names) < 2:
        sys.exit("need 2 GPUs (GPU T4 x2 accelerator); got: " + repr(names))
    return names


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        log(f"download {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    return dest


def install_llama() -> Path:
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
    root = WORK / "llama"
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-{CUDA}-x64.tar.gz",
        f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-{CUDA}-x64.tar.gz",
    ):
        with tarfile.open(fetch(f"{base}/{name}", WORK / name)) as t:
            t.extractall(root, filter="data")
    server = next(root.rglob("llama-server"))
    libdirs = sorted({str(p.parent) for p in root.rglob("*.so*")})
    os.environ["LD_LIBRARY_PATH"] = ":".join([*libdirs, os.environ.get("LD_LIBRARY_PATH", "")])
    probe = sh(f"{server} --list-devices 2>&1", check=False)
    log(f"prebuilt {LLAMA_TAG}: {probe.strip()[-400:]}")
    if probe.count("CUDA") >= 2:
        return server
    log("prebuilt binary does not see 2 CUDA devices: building from source (sm_75)")
    src = WORK / "llama.cpp"
    sh(
        f"git clone --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp {src}"
        f" && cd {src} && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75"
        " -DLLAMA_CURL=OFF && cmake --build build -j --target llama-server"
    )
    return src / "build" / "bin" / "llama-server"


def download_model() -> Path:
    free_gib = shutil.disk_usage(WORK).free / 2**30
    name = GGUF if free_gib >= 30 else GGUF_FALLBACK
    log(f"disk free {free_gib:.0f} GiB -> {name}")
    sh(f"{sys.executable} -m pip install -q -U 'huggingface_hub[hf_xet]'", check=False)
    from huggingface_hub import hf_hub_download

    t = time.monotonic()
    path = Path(hf_hub_download(HF_REPO, name, local_dir=str(WORK / "models")))
    log(f"model ready in {time.monotonic() - t:.0f}s ({path.stat().st_size / 2**30:.1f} GiB)")
    return path


def get(path: str, key: str, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.status, r.read().decode()
    except Exception as e:
        return getattr(e, "code", 0) or 0, ""


def start_server(server: Path, model: Path, key: str) -> subprocess.Popen[bytes]:
    args = [
        str(server), "-m", str(model),
        "--jinja", "-c", "8192", "-ngl", "999", "-ts", "1,1", "-np", "2",
        "--reasoning-budget", "0",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--api-key", key, "--alias", "evon", "--metrics",
        "--host", "127.0.0.1", "--port", str(PORT),
    ]  # fmt: skip
    logf = open(WORK / "llama-server.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(  # noqa: S603
        args,
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    while True:
        if proc.poll() is not None:
            tail = (WORK / "llama-server.log").read_text(errors="replace")[-3000:]
            sys.exit(f"llama-server exited {proc.returncode}:\n{tail}")
        code, body = get("/health", key)
        if code == 200 and '"ok"' in body:
            log("llama-server healthy")
            return proc
        time.sleep(5)


def start_tunnel() -> tuple[subprocess.Popen[bytes], str]:
    cf = fetch(
        f"https://github.com/cloudflare/cloudflared/releases/download/{CLOUDFLARED}/"
        "cloudflared-linux-amd64",
        WORK / "cloudflared",
    )
    cf.chmod(0o755)
    logp = WORK / "cloudflared.log"
    logp.write_text("")
    proc = subprocess.Popen(  # noqa: S603
        [str(cf), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=open(logp, "wb"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
    )
    import re

    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for _ in range(120):
        m = pat.search(logp.read_text(errors="replace"))
        if m:
            url = m.group(0)
            for _ in range(60):  # wait until the public hostname answers
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=5) as r:  # noqa: S310
                        if r.status == 200:
                            return proc, url
                except Exception:  # noqa: S110
                    pass
                time.sleep(2)
            return proc, url
        time.sleep(1)
    sys.exit("cloudflared gave no URL:\n" + logp.read_text(errors="replace")[-2000:])


def completions(key: str) -> float:
    """llamacpp:requests_processing + tokens predicted counter from /metrics (idle detection)."""
    code, body = get("/metrics", key)
    if code != 200:
        return -1.0
    total = 0.0
    for line in body.splitlines():
        if line.startswith(("llamacpp:n_decode_total", "llamacpp:tokens_predicted_total")):
            total += float(line.split()[-1])
    return total


def stop_path(key: str) -> str:
    """``scripts/evon_kaggle.sh stop`` requests this path through the tunnel; llama-server
    answers 404 but logs the request line, which the watchdog sees. Derived from the key, so
    only a key holder can stop the session (the Kaggle API has no public cancel call)."""
    import hashlib

    return "/sevi-stop/" + hashlib.sha256(f"sevi-stop:{key}".encode()).hexdigest()[:24]


def announce(line: str, key: str) -> None:
    """Print the marker (session log) and publish it to a key-derived ntfy.sh topic, so the
    push script can find the URL even when Kaggle's live log stream lags. The topic name is a
    hash of the key; the URL alone is useless without the key."""
    import hashlib

    print(line, flush=True)
    topic = "sevi-evon-" + hashlib.sha256(f"sevi-topic:{key}".encode()).hexdigest()[:32]
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=line.encode(), method="POST"
        )
        urllib.request.urlopen(req, timeout=10).close()  # noqa: S310
    except Exception as e:
        log(f"ntfy publish failed ({type(e).__name__}); the session log still has the URL")


def stop_requested(key: str) -> bool:
    try:
        return stop_path(key) in (WORK / "llama-server.log").read_text(errors="replace")
    except OSError:
        return False


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"llama.cpp {LLAMA_TAG}; max {MAX_HOURS:g} h; idle stop after {IDLE_MIN:g} min")
    gpus()
    key = api_key()
    server = install_llama()
    model = download_model()
    srv = start_server(server, model, key)
    tun, url = start_tunnel()
    announce(f"SEVI_EVON_URL={url}", key)
    log(json.dumps({"base_url": f"{url}/v1", "model": "evon", "max_concurrency": 2}))
    deadline = T0 + MAX_HOURS * 3600
    last_count, last_active = completions(key), time.monotonic()
    while True:
        time.sleep(30)
        now = time.monotonic()
        if stop_requested(key):
            log("stop requested by the operator; stopping")
            break
        if srv.poll() is not None:
            log(f"llama-server exited {srv.returncode}; stopping")
            break
        if tun.poll() is not None:
            log("tunnel died; restarting (the URL changes)")
            tun, url = start_tunnel()
            announce(f"SEVI_EVON_URL={url}", key)
        c = completions(key)
        if c != last_count:
            last_count, last_active = c, now
        if now > deadline:
            log("hard cap reached; stopping")
            break
        if now - last_active > IDLE_MIN * 60:
            log(f"idle for {IDLE_MIN:g} min; stopping")
            break
        if int(now - T0) % 300 < 30:
            log(f"alive; idle {now - last_active:.0f}s")
    announce("SEVI_EVON_STOPPED=1", key)
    for p in (tun, srv):
        p.terminate()


if __name__ == "__main__":
    main()
