"""Select copy variants by outcome variables (Person A spec §8.5; contracts §4.11).

Reads the outcome values from ``results/*`` (every file optional), selects for each copy entry the
first variant whose ``when`` all match (the ``when == {}`` default is last) and writes
``results/copy_selection.json`` (``CopySelectionFile``). There is no refusal path: every
combination of outcome values selects a variant (contracts §4.11 "No refusal path").

Reduced scope (BUILD-BRIEF §4): no panel, grid, carrier check or validation files exist, so the
derivation yields SHIP = NO_PANEL (or PRISMA_UNUSABLE if gates say so), GRID = NOT_RUN,
CARRIER = NOT_RUN, LP_SHIFT = WEAK and PHONE = BROWSER_ONLY.

Usage: ``uv run python scripts/select_copy.py [--copy PATH] [--results DIR] [--out PATH]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.enums import Owner
from sevi.contracts.evals import CopySelectionFile, ResultFileMeta
from sevi.core.wording import DEFAULT_COPY_PATH, REDUCED_SCOPE_OUTCOMES, parse_copy, select_all

LP_SHIFT_OK_DB = 2.0


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def outcome_values(results_dir: Path) -> dict[str, str]:
    """Derive every OutcomeVar from the results files (§8.5); missing files give the defaults."""
    gates = _load(results_dir / "gates.json") or {}
    panel = _load(results_dir / "panel_outcome.json")
    fit = _load(results_dir / "listener_fit.json")
    carrier = _load(results_dir / "channel_check.json")
    out = dict(REDUCED_SCOPE_OUTCOMES)

    # SHIP, first match
    trace = " ".join(str(x) for x in (panel or {}).get("rule_trace", ()))
    if gates.get("g1b") == "FAIL":
        out["SHIP"] = "PRISMA_UNUSABLE"
    elif panel is None or "n_listeners<6" in trace or "deadline" in trace:
        out["SHIP"] = "NO_PANEL"
    else:
        out["SHIP"] = str(panel.get("ship", "RMS"))

    # GRID (v3.1)
    no_grid = fit is None or fit.get("stop_reason") == "NO_GRID"
    no_balanced = panel is not None and "BALANCED" not in (panel.get("panel_bank_hashes") or {})
    out["GRID"] = "NOT_RUN" if no_grid or no_balanced else "RAN"

    # CARRIER
    if carrier is None or not carrier.get("rows"):
        out["CARRIER"] = "NOT_RUN"
    else:
        out["CARRIER"] = str(carrier.get("carrier", "NOT_RUN"))

    # PHONE: OK only after the G3T gate passed
    out["PHONE"] = "OK" if gates.get("g3t") == "PASS" else "BROWSER_ONLY"

    # LP_SHIFT: OK iff an observed shift >= 2.0 dB exists, else WEAK
    shift = gates.get("lp_shift_db")
    out["LP_SHIFT"] = "OK" if isinstance(shift, int | float) and shift >= LP_SHIFT_OK_DB else "WEAK"

    for var, key in (("EVON", "evon"), ("FRIENDS", "friends"), ("VERSION", "version")):
        if isinstance(gates.get(key), str):
            out[var] = gates[key]
    return out


def _git_sha() -> str:
    try:
        sha = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40
    return sha if len(sha) == 40 else "0" * 40


def build_selection(copy_path: Path, results_dir: Path) -> CopySelectionFile:
    raw = copy_path.read_bytes()
    copy = parse_copy(raw.decode("utf-8"))
    values = outcome_values(results_dir)
    return CopySelectionFile(
        meta=ResultFileMeta(
            schema_id="sevi.results.copy_selection/1",
            generated_at=datetime.now(UTC),
            git_sha=_git_sha(),
            generator="uv run python scripts/select_copy.py",
            owner=Owner.A,
            contracts_version=CONTRACTS_VERSION,
        ),
        outcome_values=values,
        selected=select_all(copy, values),
        copy_sha256=hashlib.sha256(raw).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copy", default=DEFAULT_COPY_PATH)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/copy_selection.json")
    args = ap.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "copy_selection.json"
    sel = build_selection(Path(args.copy), results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sel.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(sel.selected)} keys, outcomes {sel.outcome_values}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
