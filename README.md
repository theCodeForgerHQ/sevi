# Sevi

காது சோதனை

Sevi is a digits-in-noise hearing screen. It plays spoken digit triplets in background noise over a
phone-quality channel, adapts the noise level to the listener's answers, and reports a speech
reception threshold with plain-language guidance.

- **Tamil** is the banded language: the result maps to a band with pre-written wording.
- **English** is experimental: it reports no band.
- **Sevi is a screening, not a diagnosis.** A result that needs follow-up points
  to a professional check.

## Architecture

| layer | path | what it does |
|---|---|---|
| core measurement | `src/sevi/core/` | DSP chain, triplet generation, stimulus bank, adaptive staircase, scoring and lexicons, result and wording, the reply guard |
| calibration | `src/sevi/calibration/` | RMS-equalised bank rendering, noise, bank freeze |
| copy | `src/sevi/copy/copy.yaml` | every user-facing string, in Tamil and English |
| adapters | `src/sevi/adapters/` | Gnani Prisma (streaming + REST STT), Timbre (TTS), Evon (LLM explainer), Twilio REST; all metered by the credit ledger, behind a breaker and a per-key semaphore |
| transport | `src/sevi/transport/` | browser audio uplink and downlink |
| app | `src/sevi/app/` | FastAPI app: orchestrator, admission, store, retention, telemetry, operator API, routes |
| web | `web/static/` | plain HTML/JS client (no build step), AudioWorklet |

### Audio chain

- **Phone-quality stimulus.** Stimuli are mixed at 16 kHz, band-passed 300-3400 Hz (Butterworth
  order 4) and sent on the wire as 8 kHz μ-law (`src/sevi/core/dsp.py`,
  `src/sevi/contracts/constants.py`).
- **Speech recognition.** Spoken answers stream to Prisma v3, which runs at 16 kHz; the stream
  session upsamples 8 kHz input 2x before sending (`src/sevi/adapters/prisma_stream.py`).
- **Prompts.** Spoken instructions are pre-rendered with Timbre (`scripts/render_prompts.py`) and
  shipped under `stimuli/`.
- **Evon explainer.** After the test, Evon may add a short explanation of the band. Its reply passes
  a guard (`src/sevi/core/guard.py`) that rejects any number, medical claim or reassurance; on
  rejection, timeout, an open breaker or when disabled, Sevi falls back to the pre-written template
  (`src/sevi/app/explain.py`).

## How to run

```sh
make dev      # install deps and pre-commit hooks
make demo     # full local run with no keys: fake Prisma and Evon, synthetic test bank
make check    # ruff, ruff format --check, mypy --strict, pytest (no live calls)
```

`make demo` serves on http://127.0.0.1:8000/sevi/ (use 127.0.0.1; the page origin must match).

Real use loads secrets only through Infisical; nothing secret is stored in the repo:

```sh
infisical run --env=dev -- make serve   # public :8000 + operator :8001
make evon-up                            # Evon on a Kaggle session; writes var/run/evon.env
make evon-stop                          # stop it after use
make ingress                            # Tailscale Funnel (default); MODE=quick for Cloudflare
```

Restart `make serve` after `make evon-up` or `make ingress` so it picks up the files in `var/run/`.
The operator listener is never exposed through ingress. See `ops/ingress.md`.

## Privacy

From the consent screen (`prompt.consent_public` in `src/sevi/copy/copy.yaml`):

- In voice mode, **Gnani** processes spoken answers to turn them into text.
- **Tailscale** relays the encrypted connection and cannot read it.
- **Cloudflare** carries the connection only while the backup link is on.
- **GitHub** hosts the page.
- Sevi deletes its copy of the answers after **72 hours** (`src/sevi/app/retention.py`; logs too).

Logs and telemetry are redacted before they are written (`src/sevi/app/logging.py`,
`src/sevi/app/telemetry.py`). Evon sees only the band and the pre-written wording: no transcripts,
digits, scores or names.

## Credit safety

- Every Gnani call goes through `src/sevi/adapters/ledger.py`, which **refuses** the call
  (`BudgetExceeded`) once the per-key budget `SEVI_LEDGER__BUDGET_CREDITS_PER_KEY` is reached.
- Tests that hit real APIs are marked `@pytest.mark.live`, must declare
  `@pytest.mark.max_credits(...)`, and are skipped unless `SEVI_ALLOW_LIVE=1` (`tests/conftest.py`).
  `make check` and CI never run them and never hold secrets.

## Results

_Placeholder for benchmark numbers (#62). Nothing here yet._

## Who built what

- **Person A (NithishaVenkatesh):** measurement core, calibration, stimulus banks, copy and
  wording, Timbre prompts.
- **Person B (theCodeForgerHQ):** adapters, transport, app, web client, ops.
