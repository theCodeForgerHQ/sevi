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

All numbers below come from the repo's own tools or from live runs on the dev key; nothing is
estimated.

### Transcript scoring accuracy

`uv run python -m sevi.calibration.scoring_accuracy --report` (labelled transcript forms, scored by
the shipped lexicons):

| Lang | n | Exact | Digit precision | Digit recall |
|---|---:|---:|---:|---:|
| ta | 274 | 91 % | 98 % | 95 % |
| en | 202 | 88 % | 98 % | 94 % |

Every form is at 100 % except two known gaps, and both fail safe: the answer is marked wrong or
gets no digits, never wrongly marked correct.

- **Extra homophone** ("I want to say 8 1 5"): "to" is read as a digit, so exact is 0 % and
  digit precision is 75 %; recall stays 100 %.
- **Hindi digit words** ("ek do teen"): these aren't in either lexicon, so recall is 0 % and no
  digits are scored.

### Prisma v3 as the listener (live benchmark)

This was a live run on the dev key. A Prisma v3 stream went through the real adapter, using the
shipped frozen banks and the phone chain, with 6 triplets per cell
(`python -m sevi.calibration.bench_prisma --report`):

| Lang | SNR (dB) | n | Triplet correct | Digit correct | CAPTURE_MISS | Misrecognised | Prisma server latency p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| en | +4 | 6 | 83 % | 94 % | 0 | 1 | 80.5 / 98.8 |
| en | 0 | 6 | 17 % | 17 % | 5 | 0 | 93.0 / 93.0 |
| en | -4 | 6 | 0 % | 0 % | 6 | 0 | - / - |
| en | -8 | 6 | 0 % | 0 % | 6 | 0 | - / - |
| en | -12 | 6 | 0 % | 0 % | 6 | 0 | - / - |
| ta | +4 | 6 | 83 % | 94 % | 0 | 1 | 104.5 / 108.8 |
| ta | 0 | 6 | 17 % | 17 % | 4 | 1 | 76.0 / 76.0 |
| ta | -4 | 6 | 0 % | 0 % | 6 | 0 | - / - |
| ta | -8 | 6 | 0 % | 0 % | 6 | 0 | - / - |
| ta | -12 | 6 | 0 % | 0 % | 6 | 0 | - / - |

- At +4 dB, both languages get 83 % of triplets and 94 % of digits right. At 0 dB, 17 %.
- At -4 dB and below, Prisma's voice-activity detection (VAD) returned no segment at all
  (CAPTURE_MISS). These cells measure VAD, not recognition.
- VAD silence setting: with `min_silence` 300 ms, 2/6 triplets were correct at +4 dB and 3
  segments came back empty. With 800 ms, 5/6 were correct and none were empty. The default is
  moving to 800 ms in a separate PR.
- The noise here is the hearing-test masker. A user answering aloud in a normal room is at a far
  higher SNR, so these figures are a lower bound for capturing answers.
- n = 6 per cell, so each cell is a coarse estimate.
- Latency is Prisma's per-segment `server_latency_ms`. This run predates the wall-clock answer
  latency (last speech frame sent to final transcript received).

### Evon (explanations)

On the live guard, 1/10 explanations passed before #81/#82 and 19/19 after. The 19 were fresh
cases and every text was read by hand. p50 latency is about 1.6 s.

### Stimulus banks

The Tamil and English digit banks were rendered with Timbre v3 (voice Chetana), then frozen and
hash-pinned.

## Who built what

- **Person A (NithishaVenkatesh):** measurement core, calibration, stimulus banks, copy and
  wording, Timbre prompts.
- **Person B (theCodeForgerHQ):** adapters, transport, app, web client, ops.
