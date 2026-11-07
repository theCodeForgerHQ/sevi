"""G4 check against a running Evon endpoint (issue #73; B spec §5.6.1 host choice, VT-EVN-05).

Sends N (default 10) EXPLAIN_SPOKEN requests, exactly as ``adapters/evon.py`` builds them
(contracts §8.5.1: thinking off, json_schema response format), with the real system prompt and
code-selected spoken wordings from ``src/sevi/copy/copy.yaml`` across bands and both languages.
Reports:

- schema-valid on the first try (gate: >= 9/10, i.e. >= 90 %);
- generation tokens/s p50 (llama-server ``timings.predicted_per_second``; gate >= 12);
- end-to-end latency p50/p95 (gate p95 <= 3,000 ms);
- guard pass rate (``core.guard.check``, the same check ``app/explain.py`` applies).

Reads SEVI_EVON__BASE_URL / SEVI_EVON__API_KEY from the environment, then var/run/evon.env
(written by ``scripts/evon_kaggle.sh url``). The key is never printed. Self-hosted model: no
Gnani credits are spent. Exit 0 iff every gate passes.

    uv run python scripts/evon_g4.py [-n 10] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from sevi.adapters.evon import EvonClient, EvonOptions, check_schema
from sevi.app.explain import ALLOWED_KEYS, SYSTEM_KEY, CopyTable, explain_schema
from sevi.contracts.adapters import LlmRequest
from sevi.contracts.enums import Lang
from sevi.core import guard

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "src" / "sevi" / "copy" / "copy.yaml"
ENV_FILE = ROOT / "var" / "run" / "evon.env"

# (band, spoken wording key): one per eligible band family, alternating languages.
CASES: tuple[tuple[str, str], ...] = (
    ("PASS", "wording.W_PASS.spoken"),
    ("PASS_WITH_NOTE", "wording.W_PASS_NOTE.spoken"),
    ("BORDERLINE", "wording.W_BORDER.spoken"),
    ("REFER", "wording.W_REF_SRT.spoken"),
    ("REFER", "wording.W_REF_LOUD.spoken"),
    ("NO_BAND", "wording.W_NOBAND_RANGE.spoken"),
    ("INCONCLUSIVE", "wording.W_INCONC.spoken"),
)
GATE_SCHEMA = 0.9
GATE_TPS_P50 = 12.0
GATE_P95_MS = 3000.0


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); nan for an empty list."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, -(-len(s) * p // 100) - 1))
    return s[int(k)]


@dataclass
class Report:
    n: int = 0
    schema_ok: int = 0
    guard_ok: int = 0
    http_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    tps: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        rate = self.schema_ok / self.n if self.n else 0.0
        tps50 = statistics.median(self.tps) if self.tps else float("nan")
        p95 = pct(self.latencies_ms, 95)
        return {
            "requests": self.n,
            "schema_valid_first_try": f"{self.schema_ok}/{self.n}",
            "guard_pass": f"{self.guard_ok}/{self.n}",
            "http_errors": self.http_errors,
            "tokens_per_s_p50": round(tps50, 1),
            "latency_ms_p50": round(pct(self.latencies_ms, 50)),
            "latency_ms_p95": round(p95),
            "gates": {
                "schema>=90%": rate >= GATE_SCHEMA,
                "tps_p50>=12": tps50 >= GATE_TPS_P50,
                "p95<=3000ms": p95 <= GATE_P95_MS,
            },
            "failures": self.failures,
        }


def run(
    base_url: str,
    api_key: str | None,
    n: int,
    timeout_s: float,
    style: str = "llamacpp",
    pattern: bool = True,
    system_override: dict[Lang, str] | None = None,
    samples: list[dict[str, Any]] | None = None,
    offset: int = 0,
) -> Report:
    table = CopyTable.load(COPY)
    opts = EvonOptions(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout_s=timeout_s,
        response_format_style="openai" if style == "openai" else "llamacpp",
    )
    client = EvonClient(opts, None, None, None, None)  # type: ignore[arg-type]  # body only
    rep = Report()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    allowed_by_lang = {
        lang: [t for k in ALLOWED_KEYS if (t := table.text(k, lang))] for lang in Lang
    }
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0)) as http:
        for i in range(offset, offset + n):
            band, key = CASES[i % len(CASES)]
            lang = Lang.TA if i % 2 == 0 else Lang.EN
            system, wording = table.text(SYSTEM_KEY, lang), table.text(key, lang)
            if system_override and lang in system_override:
                system = system_override[lang]
            if not system or not wording:
                raise SystemExit(f"copy missing {SYSTEM_KEY} or {key} for {lang}")
            user = json.dumps({"band": band, "wording": wording}, ensure_ascii=False)
            schema = explain_schema(300, pattern)
            body = client.build_body(
                LlmRequest(
                    purpose="EXPLAIN_SPOKEN",
                    lang=lang,
                    system=system,
                    user=user,
                    json_schema=schema,
                    max_tokens=256,
                    temperature=0.6,
                )
            )
            rep.n += 1
            tag = f"#{i + 1} {band}/{lang.value}"
            t0 = time.perf_counter()
            try:
                r = http.post(f"{opts.base_url}/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                rep.http_errors += 1
                rep.failures.append(f"{tag}: {type(e).__name__}")
                continue
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                rep.http_errors += 1
                rep.failures.append(f"{tag}: HTTP {r.status_code}")
                continue
            rep.latencies_ms.append(ms)
            payload = r.json()
            tps = (payload.get("timings") or {}).get("predicted_per_second")
            if isinstance(tps, int | float):
                rep.tps.append(float(tps))
            choice = payload["choices"][0]
            content = str(choice["message"].get("content") or "")
            try:
                parsed = json.loads(content)
                why = check_schema(parsed, schema)
            except ValueError:
                parsed, why = None, "not JSON"
            if choice.get("finish_reason") == "length":
                why = why or "finish_reason=length"
            if why is not None:
                rep.failures.append(f"{tag}: schema {why}")
                continue
            rep.schema_ok += 1
            assert isinstance(parsed, dict)
            verdict = guard.check(
                parsed["text"],
                lang,
                "EXPLAIN_SPOKEN",
                banned=table.banned,
                allowed=[*allowed_by_lang[lang], wording],
            )
            if samples is not None:
                samples.append(
                    {
                        "tag": tag,
                        "text": parsed["text"],
                        "reasons": [x.value for x in verdict.reasons],
                    }
                )
            if verdict.ok:
                rep.guard_ok += 1
            else:
                rep.failures.append(f"{tag}: guard {[x.value for x in verdict.reasons]}")
            print(f"{tag}: {ms:.0f} ms, {tps} tok/s, guard={'ok' if verdict.ok else 'REJECT'}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-pattern", action="store_true", help="schema without the digit pattern")
    ap.add_argument("--style", choices=["llamacpp", "openai"], default=None)
    ap.add_argument("--dump", type=Path, default=None, help="write every text + guard reasons")
    ap.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="YAML {en: ..., ta: ...} system prompt to try instead of the copy table",
    )
    ap.add_argument("--offset", type=int, default=0, help="start case index (fresh samples)")
    a = ap.parse_args(argv)
    env = {**load_env_file(ENV_FILE), **{k: v for k, v in os.environ.items() if v}}
    base = env.get("SEVI_EVON__BASE_URL")
    if not base:
        print("SEVI_EVON__BASE_URL not set (run scripts/evon_kaggle.sh url)", file=sys.stderr)
        return 2
    style = a.style or env.get("SEVI_EVON__RESPONSE_FORMAT_STYLE", "llamacpp")
    override: dict[Lang, str] | None = None
    if a.prompt_file is not None:
        import yaml

        doc = yaml.safe_load(a.prompt_file.read_text(encoding="utf-8"))
        override = {Lang(k): str(v) for k, v in doc.items()}
    samples: list[dict[str, Any]] = []
    rep = run(
        base,
        env.get("SEVI_EVON__API_KEY"),
        a.n,
        a.timeout,
        style,
        not a.no_pattern,
        system_override=override,
        samples=samples,
        offset=a.offset,
    )
    if a.dump is not None:
        a.dump.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"response_format style: {style}")
    s = rep.summary()
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print(f"G4 against {base.split('//')[-1].split('/')[0]}")
        for k, v in s.items():
            if k not in ("gates", "failures"):
                print(f"  {k}: {v}")
        for k, ok in s["gates"].items():
            print(f"  gate {k}: {'PASS' if ok else 'FAIL'}")
        for f in s["failures"]:
            print(f"  - {f}")
    return 0 if all(s["gates"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
