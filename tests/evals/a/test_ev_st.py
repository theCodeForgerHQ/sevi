"""EV-ST-01 staircase simulation (Person A spec §6.6, EXP-3 part 1). Deterministic, seeded."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from sevi.contracts.procedure import ProcedureConfig
from sevi.contracts.rng import derive_seed
from sevi.core.staircase import PROCEDURE
from sevi.fakes.scenarios.staircase_driver import make_config, result_for

SLOPES = (0.12, 0.15, 0.20, 0.25)
THETAS = (-12.0, -8.0, -4.0)
REPORT_ONLY_SLOPE = 0.10
N_RUNS = int(os.environ.get("SEVI_EVST_RUNS", "1000"))


def simulate_run(slope: float, theta: float, run: int) -> float:
    scope = f"stsim:{slope:g}:{theta:g}:{run}"
    cfg: ProcedureConfig = make_config(scope, skip_volume_and_practice=True)
    rng = np.random.Generator(np.random.PCG64(derive_seed(scope, "listener")))
    s = PROCEDURE.start(cfg)
    while s.pending is not None:
        p = 1.0 / (1.0 + math.exp(-4.0 * slope * (s.pending.snr_db - theta)))
        s = PROCEDURE.record(s, result_for(s.pending, "C" if rng.random() < p else "W"))
    srt = PROCEDURE.summarize(s).srt_db
    assert srt is not None
    return srt


def run_cell(slope: float, theta: float, n_runs: int) -> dict[str, float]:
    srts = np.array([simulate_run(slope, theta, r) for r in range(n_runs)])
    return {
        "slope": slope,
        "theta_db": theta,
        "n_runs": n_runs,
        "mean_srt_db": float(srts.mean()),
        "bias_db": float(srts.mean() - theta),
        "sd_db": float(srts.std(ddof=1)),
    }


def run_ev_st_01(n_runs: int = N_RUNS) -> dict[str, object]:
    cells = [run_cell(s, t, n_runs) for s in SLOPES for t in THETAS]
    reported = [run_cell(REPORT_ONLY_SLOPE, t, n_runs) for t in THETAS]
    max_bias = max(abs(c["bias_db"]) for c in cells)
    max_sd = max(c["sd_db"] for c in cells)
    return {
        "eval": "EV-ST-01",
        "cells": cells,
        "report_only_cells": reported,
        "max_abs_bias_db": max_bias,
        "max_sd_db": max_sd,
        "passed": bool(max_bias <= 0.5 and max_sd <= 1.2),
    }


@pytest.mark.slow
def test_EV_ST_01_staircase_simulation(tmp_path: Path) -> None:
    out = run_ev_st_01()
    dest = Path(os.environ.get("SEVI_RESULTS_DIR", tmp_path)) / "staircase_sim.json"
    dest.write_text(json.dumps(out, indent=2))
    assert out["passed"] is True, out


def test_EV_ST_01_is_deterministic() -> None:
    assert simulate_run(0.15, -8.0, 3) == simulate_run(0.15, -8.0, 3)
