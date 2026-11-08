<div align="center">

# Sevi

### காது சோதனை · a digits-in-noise hearing screen over a phone-quality channel

![measure](https://img.shields.io/badge/measure-digits--in--noise_SRT-2f8f4e)
![lang](https://img.shields.io/badge/banded-Tamil-c0562f)
![lang-en](https://img.shields.io/badge/English-experimental,_no_band-607d8b)
![stt](https://img.shields.io/badge/STT-Gnani_Prisma_v3-3b6ea5)
![backend](https://img.shields.io/badge/backend-FastAPI_·_Python-15212e)
![types](https://img.shields.io/badge/mypy-strict-2f8f4e)
![screening](https://img.shields.io/badge/screening-not_a_diagnosis-c0392b)

</div>

> Sevi plays spoken digit triplets in background noise over a phone-quality channel, adapts the
> noise level to the listener's answers, and reports a speech reception threshold (SRT) with
> plain-language guidance. Answers can be spoken (streamed to Gnani Prisma v3) or typed on a
> keypad. **Sevi is a screening, not a diagnosis.** A result that needs follow-up points to a
> professional check.

---

## Why it matters

Hearing loss is common, gradual and easy to miss, and a hearing check usually needs a booth, an
audiometer and someone trained to run it. A digits-in-noise test needs none of those: it asks
the listener to repeat three digits heard in noise, and how much noise they can tolerate is
itself the measurement. Sevi runs that test in the listener's language (Tamil first) on the
devices people already have, and keeps every decision that matters in code, not in a model.

| | What | Source of truth |
|---|---|---|
| **Procedure** | 1-up/1-down staircase, 24 scored triplets, start 0 dB SNR, steps 4 dB then 2 dB, clamped to -20..+10 dB | `src/sevi/core/staircase.py`, `src/sevi/contracts/constants.py` |
| **SRT** | mean SNR of scored trials 5..24 plus the virtual 25th (21 values) | `Staircase.summarize` |
| **Result** | a first-match truth table, rules T1..T15 (+T10b), 10 bands | `src/sevi/core/result.py` |
| **Wording** | every user-facing string pre-written in Tamil and English | `src/sevi/copy/copy.yaml` |
| **Audio** | 16 kHz mix, band-pass 300-3400 Hz, 8 kHz μ-law on the wire | `src/sevi/core/dsp.py` |
| **Stimuli** | Timbre-rendered digit banks, RMS-equalised, frozen and hash-pinned | `stimuli/`, `src/sevi/calibration/freeze.py` |
| **Scoring** | lexicon normaliser over transcripts and keypad digits; ta 91 % / en 88 % exact on labelled forms | `src/sevi/core/scoring.py` |
| **LLM** | Evon may rephrase the code-selected wording; a guard rejects numbers, medical claims, reassurance | `src/sevi/app/explain.py`, `src/sevi/core/guard.py` |
| **Credits** | every Gnani call reserved against a hard per-key budget before any network I/O | `src/sevi/adapters/ledger.py` |
| **Privacy** | answers, sessions and logs deleted after 72 hours | `src/sevi/app/retention.py` |

## Table of contents

1. [What it is](#1--what-it-is)
2. [Architecture at a glance](#2--architecture-at-a-glance)
3. [The measurement procedure](#3--the-measurement-procedure)
4. [Scoring an answer](#4--scoring-an-answer)
5. [From SRT to a band](#5--from-srt-to-a-band)
6. [Stimuli and calibration](#6--stimuli-and-calibration)
7. [One browser voice session](#7--one-browser-voice-session)
8. [Session state machine](#8--session-state-machine)
9. [Admission and credit safety](#9--admission-and-credit-safety)
10. [The credit ledger](#10--the-credit-ledger)
11. [Evon explanation, guard and fallback](#11--evon-explanation-guard-and-fallback)
12. [Phone (Twilio) flow](#12--phone-twilio-flow)
13. [Privacy and data](#13--privacy-and-data)
14. [Results](#14--results)
15. [Limitations](#15--limitations)
16. [Deployment](#16--deployment)
17. [Getting started](#17--getting-started)
18. [Project structure](#18--project-structure)
19. [Engineering discipline](#19--engineering-discipline)
20. [Who built what](#20--who-built-what)

---

## 1 · What it is

- **Tamil** is the banded language: the result maps to a band with pre-written wording.
- **English** is experimental: it reports no band.
- **Sevi is a screening, not a diagnosis.** A result that needs follow-up points
  to a professional check.

Two channels run the *same* test (one adaptive procedure, the same bank renders, the same scorer
and the same result classifier); only the transport differs:

| channel | answers | transport |
|---|---|---|
| **Browser** | voice (Prisma v3 streaming STT) or keypad | plain HTML/JS client, WebSocket `/web/stream`, AudioWorklet |
| **Phone** (Twilio trial, demo only, off by default) | keypad (DTMF); optional recording + Prisma REST | Twilio webhooks `/twilio/*`, `<Play>` + `<Gather>` |

### Audio chain

- **Phone-quality stimulus.** Stimuli are mixed at 16 kHz, band-passed 300-3400 Hz (Butterworth
  order 4) and sent on the wire as 8 kHz μ-law (`src/sevi/core/dsp.py`,
  `src/sevi/contracts/constants.py`).
- **Speech recognition.** Spoken answers stream to Prisma v3, which runs at 16 kHz; the stream
  session upsamples 8 kHz input 2x before sending (`src/sevi/adapters/prisma_stream.py`). A
  stream carries exactly one locale (live v3 rejects a list of locales).
- **Prompts.** Spoken instructions are pre-rendered with Timbre (`scripts/render_prompts.py`) and
  shipped under `stimuli/`.
- **Evon explainer.** After the test, Evon may add a short explanation of the band. Its reply passes
  a guard (`src/sevi/core/guard.py`) that rejects any number, medical claim or reassurance; on
  rejection, timeout, an open breaker or when disabled, Sevi falls back to the pre-written template
  (`src/sevi/app/explain.py`).

---

## 2 · Architecture at a glance

| layer | path | what it does |
|---|---|---|
| core measurement | `src/sevi/core/` | DSP chain, triplet generation, stimulus bank, adaptive staircase, scoring and lexicons, result and wording, the reply guard |
| calibration | `src/sevi/calibration/` | RMS-equalised bank rendering, noise, bank freeze |
| copy | `src/sevi/copy/copy.yaml` | every user-facing string, in Tamil and English |
| adapters | `src/sevi/adapters/` | Gnani Prisma (streaming + REST STT), Timbre (TTS), Evon (LLM explainer), Twilio REST; all metered by the credit ledger, behind a breaker and a per-key semaphore |
| transport | `src/sevi/transport/` | browser audio uplink and downlink |
| app | `src/sevi/app/` | FastAPI app: orchestrator, admission, store, retention, telemetry, operator API, routes |
| web | `web/static/` | plain HTML/JS client (no build step), AudioWorklet |

The contracts package (`src/sevi/contracts/`) holds the enums, constants, wire models and ports
both halves code against; `src/sevi/fakes/` provides fake Prisma, Evon, Timbre and Twilio so the
whole app runs with no keys.

```mermaid
flowchart LR
    subgraph Client["Clients"]
        WEB["web/static<br/>HTML/JS + AudioWorklet"]
        PHONE["Phone via Twilio"]
    end
    subgraph App["src/sevi/app (FastAPI)"]
        ROUTES["routes<br/>public · web_ws · report · twilio · audio · operator"]
        ADM["admission.py<br/>levels L0-L4, queue, per-IP limits"]
        ORCH["orchestrator.py<br/>BrowserSession FSM"]
        IVR["ivr.py<br/>phone step machine"]
        EXPL["explain.py"]
        STORE[("store.py<br/>SQLite")]
        RET["retention.py<br/>72 h"]
    end
    subgraph Core["src/sevi/core (pure)"]
        STAIR["staircase.py"]
        SCORE["scoring.py + lexicon"]
        RES["result.py + wording.py"]
        BANK["bank.py + stimulus.py + dsp.py"]
        GUARD["guard.py"]
    end
    subgraph Adapters["src/sevi/adapters"]
        LEDGER["ledger.py"]
        BRK["breaker.py + key_semaphore.py"]
        PRS["prisma_stream.py / prisma_rest.py"]
        EVON["evon.py"]
        TW["twilio_rest.py"]
        TIM["timbre.py"]
    end
    TRANS["src/sevi/transport<br/>web_ws.py · uplink.py"]

    WEB --> ROUTES
    PHONE --> ROUTES
    ROUTES --> ADM --> ORCH
    ROUTES --> IVR
    ROUTES --> TRANS --> ORCH
    ORCH --> STAIR
    ORCH --> SCORE
    ORCH --> RES
    ORCH --> BANK
    IVR --> STAIR
    IVR --> SCORE
    IVR --> RES
    ORCH --> EXPL --> GUARD
    EXPL --> EVON
    ORCH --> PRS
    IVR --> TW
    PRS --> LEDGER
    TIM --> LEDGER
    PRS --> BRK
    EVON --> BRK
    ORCH --> STORE
    IVR --> STORE
    LEDGER --> STORE
    RET --> STORE
```

---

## 3 · The measurement procedure

The procedure is a pure function of its state (`src/sevi/core/staircase.py`): every answer
returns a new `ProcedureState`, and `_next_snr` is the only SNR rule. All numbers below are
constants in `src/sevi/contracts/constants.py`.

| phase | what happens | pass rule |
|---|---|---|
| **VOLUME** | 2 triplets at +10 dB SNR, up to 2 rounds (a "turn it up" prompt between) | >= 1 of 2 correct; else `VOLUME_FAILED` |
| **PRACTICE** | 3 triplets at +6, +2, -2 dB, up to 2 rounds | >= 2 of 3 correct; else `PRACTICE_FAILED` |
| **SCORED** | 24 triplets, start 0 dB; correct -> SNR down, wrong -> SNR up | step 4 dB until the first wrong scored trial, then 2 dB; clamped to -20..+10 dB |
| **LOUD** | after 3 no-responses in a row: 2 triplets at +10 dB, not scored, SNR unchanged | 2 wrong loud triplets across the session -> `LOUD_FAILED`; a 3rd no-response run after 2 loud checks -> `NO_RESPONSE_END` |

A **ceiling stop** ends the test early after 6 consecutive valid wrong answers at +10 dB
(`CEILING_STOP`). A `CAPTURE_MISS` (the recogniser heard nothing) replays the same presentation
once before it counts.

```mermaid
flowchart TD
    START(["start"]) --> VOL["VOLUME<br/>2 triplets at +10 dB"]
    VOL -->|"at least 1 of 2 correct"| PRA["PRACTICE<br/>+6, +2, -2 dB"]
    VOL -->|"fail, round 1"| VOL
    VOL -->|"fail, round 2"| T1(["VOLUME_FAILED"])
    PRA -->|"at least 2 of 3 correct"| SC["SCORED trial n<br/>at current SNR"]
    PRA -->|"fail, round 1"| PRA
    PRA -->|"fail, round 2"| T2(["PRACTICE_FAILED"])
    SC --> ANS{"answer"}
    ANS -->|"correct"| DOWN["SNR down<br/>4 dB, or 2 dB after the first wrong"]
    ANS -->|"wrong"| UP["SNR up<br/>4 dB, or 2 dB after the first wrong"]
    DOWN --> CLAMP["clamp to -20..+10 dB"]
    UP --> CLAMP
    CLAMP --> CEIL{"6 valid wrong in a row at +10 dB?"}
    CEIL -->|"yes"| T5(["CEILING_STOP"])
    CEIL -->|"no"| DONE{"24 scored?"}
    DONE -->|"yes"| T6(["COMPLETE<br/>SRT = mean SNR of trials 5..24 + virtual 25th"])
    DONE -->|"no"| NR{"3 no-responses in a row?"}
    NR -->|"no"| SC
    NR -->|"yes, fewer than 2 loud checks"| LOUD["LOUD check<br/>2 triplets at +10 dB, not scored"]
    NR -->|"yes, 2 loud checks done"| T7(["NO_RESPONSE_END"])
    LOUD -->|"2 wrong loud in total"| T3(["LOUD_FAILED"])
    LOUD -->|"heard"| SC
```

`summarize` also records **censoring**: `HIGH` if any trial from 5 on was wrong at +10 dB, `LOW`
if any was correct at -20 dB, and the number of reversals; both feed the banding rules below.

---

## 4 · Scoring an answer

Voice transcripts and keypad digits both go through `LexiconScorer.score`
(`src/sevi/core/scoring.py`): a lexicon normaliser (`src/sevi/core/lexicon/`), Levenshtein
alignment and the status rules. Correctness is separate from status; a triplet is correct only
when all three digits match.

```mermaid
flowchart TD
    IN["transcript or DTMF digits"] --> NORM["normalise<br/>lexicon words to digits, intents"]
    NORM --> Q{"what was heard?"}
    Q -->|"exactly 3 digits"| OK["OK<br/>scored correct or wrong"]
    Q -->|"1-2 digits"| PART["PARTIAL<br/>wrong, valid"]
    Q -->|"more than 3 digits"| AMB["AMBIGUOUS<br/>re-ask once, then wrong valid"]
    Q -->|"'don't know' intent"| DK["DONT_KNOW<br/>wrong, valid, no re-ask"]
    Q -->|"'repeat' intent"| REP["REPEAT_REQUEST<br/>replay once"]
    Q -->|"speech onset inside the gate"| EARLY["EARLY_ANSWER<br/>re-ask, not scored"]
    Q -->|"no speech energy, no key"| NRS["NO_RESPONSE<br/>wrong, valid"]
    Q -->|"recogniser returned nothing"| CM["CAPTURE_MISS<br/>replay once"]
    OK --> ACT["ScoreAction: ACCEPT"]
    PART --> ACT
    DK --> ACT
    NRS --> ACT
    AMB --> RE["ScoreAction: REASK"]
    EARLY --> RE
    REP --> RP["ScoreAction: REPLAY"]
    CM --> RP
```

The answer window closes on three digits, a settle window, a timeout, a complete DTMF entry or a
"don't know" intent. The settle window is max(`t_settle_ms`, `min_silence` + 300 ms), 1,100 ms by
default (see [Results](#14--results)).

---

## 5 · From SRT to a band

`classify` (`src/sevi/core/result.py`) is a first-match truth table. It never raises: if it
reaches the SRT rules without 24 scored trials and an SRT, it returns the T8 decision. Each
decision carries a band, a wording class from `copy.yaml`, and the surfaces it may appear on
(spoken, page, report, operator).

```mermaid
flowchart TD
    A["ResultInput<br/>SRT summary, end reason, norms, channel, mode"] --> R1{"T1 VOLUME_FAILED<br/>T2 PRACTICE_FAILED<br/>T3 LOUD_FAILED<br/>T4 can't hear<br/>T5 ceiling"}
    R1 -->|"match"| REF1(["REFER"])
    R1 -->|"no"| R6{"T6 flags above 3"}
    R6 -->|"yes"| INC(["INCONCLUSIVE"])
    R6 -->|"no"| R7{"T7 stopped early<br/>hang-up, abort, timeout, no response"}
    R7 -->|"yes"| INCO(["INCOMPLETE"])
    R7 -->|"no"| R8{"T8 server error or socket lost"}
    R8 -->|"yes"| FAIL(["FAILED"])
    R8 -->|"no"| R9{"T9 consent no-reply,<br/>dial failed, queue abandoned"}
    R9 -->|"yes"| NS(["NOT_STARTED"])
    R9 -->|"no"| R10{"T10 consent declined"}
    R10 -->|"yes"| DEC(["DECLINED"])
    R10 -->|"no"| R10B{"T10b SRT at or above 0 dB"}
    R10B -->|"yes"| REF2(["REFER<br/>absolute"])
    R10B -->|"no"| R11{"T11 no usable norm cell<br/>experimental language, speaker,<br/>mixed mode, n below 6, bank mismatch"}
    R11 -->|"yes"| NB(["NO_BAND"])
    R11 -->|"no"| R12{"T12 SRT above NH95 + 3 dB"}
    R12 -->|"yes"| REF3(["REFER"])
    R12 -->|"no"| R13{"T13 within 3 dB above NH95,<br/>v0 cut-off within 1 dB,<br/>under 4 reversals, or HIGH censor"}
    R13 -->|"yes"| BOR(["BORDERLINE"])
    R13 -->|"no"| R14{"T14 HHIE total 10 or more"}
    R14 -->|"yes"| PWN(["PASS_WITH_NOTE"])
    R14 -->|"no"| PASS(["T15 PASS"])
```

- **Norm cells.** NH95 = mean + 1.645 x sd of a normal-hearing cell keyed by language, channel,
  mode and transducer. A cell needs n >= 6 (`v0`; `v1` from n >= 10) and the same bank hash as the
  session.
- **English** always lands on T11 `EXPERIMENTAL_LANGUAGE` unless a referral rule matched first.
  **Phone** results are `NO_BAND` (T11 `NO_CELL`) unless a referral rule matches.
- **SRT display** rounds half up to 0.5 dB with a true minus sign; the number is shown only where
  the wording class allows it.

---

## 6 · Stimuli and calibration

The Tamil and English digit banks were rendered with Timbre v3 (voice Chetana), then frozen and
hash-pinned. The calibration modules are offline by default and refuse to spend credits without
an explicit `--yes`.

```mermaid
flowchart LR
    subgraph Render["calibration/render.py"]
        TTS["Timbre via the ledger-guarded adapter<br/>48 kHz wav"] --> CACHE[("write-once cache<br/>key = sha256 of request")]
        CACHE --> TRIM["resample to 16 kHz,<br/>trim, RMS-equalise"]
    end
    TRIM --> REC["stimuli/lang/digits + render.json"]
    REC --> FRZ["calibration/freeze.py<br/>re-verify digit hashes,<br/>regenerate noise.wav, LTASS check"]
    FRZ --> MAN["manifest.json<br/>SHIPPED, frozen, bank_hash"]
    MAN --> PIN["SEVI_RESULT__FROZEN_BANK_HASHES"]
    PIN --> LOAD["core/bank.py at startup<br/>verify every sha256 + bank_hash<br/>else BANK_HASH_REJECTED"]
    LOAD --> RT["render_trial<br/>mix at SNR, band-pass, 8 kHz, mu-law"]
    RT --> BENCH["calibration/bench_prisma.py<br/>real PrismaStreamClient, core.scoring"]
    BENCH --> R1["Prisma vs SNR table"]
    LEX["labelled transcript forms"] --> ACC["calibration/scoring_accuracy.py<br/>shipped lexicons"]
    ACC --> R2["scoring accuracy table"]
```

- **Bank hash.** `bank_hash` is the sha256 of the canonical (sorted-key, no-whitespace) manifest
  JSON without `bank_hash`, `frozen` and `frozen_at`, so key order never changes it.
- **Norms are tied to a bank.** A norm cell whose bank hash differs from the session's is never
  used to band (T11 `NORMS_BANK_MISMATCH`).
- **Benchmark plan is deterministic.** Triplets and noise seeds come from `derive_seed`, so a
  re-run plays the same audio.
- `make demo` uses the synthetic `stimuli/test-bank` with `SEVI_RESULT__ALLOW_UNFROZEN_BANK`.

---

## 7 · One browser voice session

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser (web/static)
    participant S as Sevi server
    participant L as Ledger
    participant P as Gnani Prisma v3
    participant E as Evon (llama-server)

    B->>S: POST /api/v1/sessions {lang, mode: voice}
    S->>S: admission checks (lang, readiness, keypad cap, voice offered, per-IP)
    alt stream permit free
        S-->>B: 201 ADMITTED {ws_url, ws_token}
    else no permit
        S-->>B: 201 QUEUED {position, eta_s}
        B->>S: GET /api/v1/sessions/{id} (poll)
    end
    B->>S: WS /web/stream?sid= then "start"
    S-->>B: ui consent
    B->>S: ui consent_given
    S->>L: reserve stream seconds (before any network I/O)
    L-->>S: ok, or BudgetExceeded
    S->>P: open stream (one locale, 16 kHz)
    S-->>B: ui mic start
    loop each presentation (volume, practice, 24 scored, loud)
        S-->>B: media (8 kHz mu-law stimulus) + mark
        B-->>S: mark (played)
        B->>S: media (mic frames, 8 kHz)
        S->>P: frames upsampled 2x, then commit
        P-->>S: transcript segments
        S->>S: score, next SNR
        S-->>B: ui feedback / phase
        S->>L: top up when billed seconds reach the reservation
    end
    S->>S: summarize SRT, classify band
    S-->>B: ui result (wording, report link)
    S->>E: band + pre-written wording only
    E-->>S: JSON text
    S->>S: guard.check, else template
    S-->>B: ui explanation
    S->>P: close
    S->>L: commit actual billed seconds
    S-->>B: ui closed
```

The browser never talks to Gnani or Evon. Only mic audio (to Prisma) and the band plus wording
(to Evon) leave the server.

---

## 8 · Session state machine

One `BrowserSession` per test (`src/sevi/app/orchestrator.py`). Every event (transport, STT,
timers, UI, internal results) goes through one queue and is handled strictly in arrival order;
every transition records its spec rule id (`S..`) and emits `fsm.transition`. The FSM talks to a
`CallTransport`, never to a socket.

```mermaid
stateDiagram-v2
    [*] --> AWAIT_START: admitted (S01 keypad, S02 voice), socket accepted (S13)
    AWAIT_START --> CONSENT: start, token ok (S11)
    AWAIT_START --> ENDED: start timeout, socket closed, abort (S14-S16)
    CONSENT --> INSTRUCTIONS: consent given (S22)
    CONSENT --> RESULT: declined (S23) or no reply (S26)
    INSTRUCTIONS --> INSTRUCTIONS: repeat (S41)
    INSTRUCTIONS --> TEST: procedure started (S42)
    TEST --> TEST: presentation loop
    TEST --> RESULT: terminal reached (S50)
    TEST --> RESULT: can't hear (S52), hang-up (S53), socket lost (S54)
    TEST --> RESULT: server error (S55), session max (S56), abort (S57)
    RESULT --> RESULT: next playlist clip (S61)
    RESULT --> CLOSING: playlist done (S62)
    CLOSING --> ENDED: closed (S70)
    ENDED --> [*]
```

- **Queueing happens before the session exists.** A queued voice request (S03) waits in the
  admission queue and is only created as a session once a stream permit frees.
- **Mic loss** in voice mode switches the session to keypad (`MIC_LOST` flag, S58/S28) instead of
  ending it.
- **Presentation sub-states** inside `TEST`: `PRE_PROMPT -> SEND -> GATED -> LISTEN -> COMMIT ->
  SCORE -> (REASK_PROMPT | FEEDBACK)`.
- **Timers are deterministic**: absolute `Clock.mono_ms()` deadlines (`T_START`,
  `T_SESSION_MAX`, `T_MIC_FIRST`, `T_EXPLAIN`, ...), so tests drive `dispatch` with a fake clock.
- **Crash path**: an unhandled error in a live session goes to `RESULT` with the T8 decision
  (S55); before the test it goes straight to `ENDED` (S18).

---

## 9 · Admission and credit safety

`Admission` (`src/sevi/app/admission.py`) computes a spike level and decides every
`POST /api/v1/sessions`:

| level | condition |
|---|---|
| **L4** | readiness down or draining |
| **L3** | browser-voice or Prisma switch off, the public key's Prisma breaker open, `public_key_low`, or `exposure_paused` |
| **L2** | keypad full (120 active keypad sessions) |
| **L1** | voice queue longer than 3 |
| **L0** | normal |

```mermaid
flowchart TD
    REQ["POST /api/v1/sessions"] --> G1{"language bank loaded?"}
    G1 -->|"no"| E1["LANG_UNAVAILABLE"]
    G1 -->|"yes"| G2{"draining or readiness down?"}
    G2 -->|"yes"| E2["SERVICE_DEGRADED, retry 60 s"]
    G2 -->|"no"| G3{"keypad full?"}
    G3 -->|"yes"| E3["ADMISSION_WAIT, waiting room"]
    G3 -->|"no"| G4{"voice?"}
    G4 -->|"yes"| VO{"voice offered?<br/>switches on, breaker closed,<br/>not public_key_low,<br/>not exposure_paused, queue ok"}
    VO -->|"no"| E4["VOICE_UNAVAILABLE"]
    VO -->|"yes"| G5
    G4 -->|"no"| G5{"trusted client IP over<br/>hourly limit (keypad 20, voice 3)?"}
    G5 -->|"yes"| E5["RATE_LIMITED_IP, retry 600 s"]
    G5 -->|"no, keypad"| AK["ADMITTED (KeyId NONE, no credits)"]
    G5 -->|"no, voice"| PERM{"stream permit on the public key?"}
    PERM -->|"yes"| AV["ADMITTED"]
    PERM -->|"no, queue below 3"| Q["QUEUED, poll"]
    PERM -->|"no, queue full"| E6["VOICE_UNAVAILABLE"]
```

Credit gates that feed `voice_offered` and L3:

- **`public_key_low`** (ledger): true after a 402 from Gnani (persisted "exhausted" flag), or when
  used credits reach total x (1 - reserve) minus the stop-session estimate, or one more voice
  session would pass `SEVI_LEDGER__BUDGET_CREDITS_PER_KEY`.
- **`exposure_paused`**: stream seconds not yet reconciled against the Gnani dashboard reach 400
  credits' worth. A manual reconcile resets it. Both are cached for 5 s.
- **Voice slot = stream permit.** A voice session holds a `KeySemaphore` stream permit on the
  public key (Key B, or Key A when B is aliased) for its whole life.

Tests that hit real APIs are marked `@pytest.mark.live`, must declare
`@pytest.mark.max_credits(...)`, and are skipped unless `SEVI_ALLOW_LIVE=1` (`tests/conftest.py`).
`make check` and CI never run them and never hold secrets.

---

## 10 · The credit ledger

Every Gnani call goes through `src/sevi/adapters/ledger.py`, which **refuses** the call
(`BudgetExceeded`) once the per-key budget `SEVI_LEDGER__BUDGET_CREDITS_PER_KEY` is reached. Calls
use `async with ledger.billed(key, purpose, stt_s=...)`:

```mermaid
stateDiagram-v2
    [*] --> Reserve: billed() entered
    Reserve --> Refused: spent + outstanding + estimate above budget, rate unset, or key exhausted
    Refused --> [*]: BudgetExceeded, call never made
    Reserve --> Reserved: row written in the same BEGIN IMMEDIATE transaction
    Reserved --> Reserved: top_up (stream billed seconds reach the reservation)
    Reserved --> Committed: block exits, or raises after actual()
    Reserved --> Released: raises after not_billed()
    Reserved --> HeldAsSpent: raises with no actual (fail closed)
    Committed --> [*]: LedgerEvent origin server, reservation deleted
    Released --> [*]: reservation deleted
    HeldAsSpent --> Settled: retention settles stale reservations
    Settled --> [*]
    Committed --> Reconciled: operator POST /op/api/ledger/reconcile (dashboard reading)
    Reconciled --> [*]
```

- **No overshoot under concurrency.** The budget check and the reservation insert share one
  SQLite write lock, so concurrent tasks or processes cannot pass the budget together.
- **Restart-safe.** Events, reservations and the 402 flag live in SQLite; reservations left by a
  crash count as spent until settled or cleared by an operator.
- **Spend of a key** = max(sum of estimates, latest dashboard reading + estimates after it) +
  outstanding reservations: a higher dashboard figure always wins.
- **Buckets.** Each purpose maps to a bucket per key: Key A `CAL` (calibration, bench, vt, g5),
  `OPS` (prompt render, validation, phone record, channel check), `RESERVE` (interview); Key B
  `DEV` (g5, vt, dev, video), `PUBLIC` (public voice), `RESERVE`. `cal_stop`, `ops_stop` and
  `dev_stop` trip at fixed fractions of the key total.
- **Twilio** trial minutes are counted per CallSid; `check_dial` refuses a dial whose worst case
  would pass the trial-minutes stop.

---

## 11 · Evon explanation, guard and fallback

Code decides the result; Evon only rephrases the code-selected wording
(`src/sevi/app/explain.py`). Evon is a self-hosted llama-server behind an OpenAI-compatible API
(`src/sevi/adapters/evon.py`): no Gnani credits, protected by its own breaker, never retried by the
adapter.

```mermaid
flowchart TD
    D["ResultDecision"] --> EL{"eligible?<br/>switch on, browser channel,<br/>breaker closed, band explainable"}
    EL -->|"no"| SK["SKIPPED"]
    EL -->|"yes"| PRE{"enabled, breaker closed,<br/>health up, semaphore within deadline?"}
    PRE -->|"no"| TPL["TEMPLATE<br/>pre-written copy"]
    PRE -->|"yes"| IN["prompt = band + pre-written wording<br/>no transcripts, digits, SRT or names"]
    IN --> CALL["complete_json, JSON schema,<br/>max 300 chars, no-digit pattern"]
    CALL -->|"timeout, breaker, unavailable"| TPL
    CALL -->|"schema invalid"| RETRY
    CALL --> G{"guard.check<br/>EMPTY, LENGTH, NUMERAL, NUMBER_WORD,<br/>UNIT, MEDICAL, SCRIPT, BANNED, REASSURANCE"}
    G -->|"ok"| LIVE["LIVE<br/>guarded text"]
    G -->|"rejected"| RETRY{"attempt 2 and at least 3 s left?"}
    RETRY -->|"yes, regenerate with rejection reasons"| CALL
    RETRY -->|"no"| TPL
```

- **Hard deadline** 10 s from the call; each request times out at 6 s or the time left.
- **The guard normalises first** (NFKC, strips zero-width characters, casefolds, folds Cyrillic and
  Greek lookalikes) so a number cannot be smuggled past it. Our own disclaimer sentences and the
  approved wording itself are allowed verbatim.
- **Phone sessions never call Evon**, and the report never carries Evon text.

---

## 12 · Phone (Twilio) flow

`src/sevi/app/ivr.py`, demo only and off by default (`SEVI_PHONE__ENABLED=false`: every
`/twilio/*` and `/audio/*` request is 404 and the operator dial is 409 `PHONE_DISABLED`).

> **Status: built and tested against a fake Twilio, not live-demonstrated.** The IVR, webhooks,
> signature checks and trial-minute ledger run in the test suite against `src/sevi/fakes/`. A
> live call needs a paid (upgraded) Twilio account: trial accounts can no longer verify a caller
> ID or buy a number. The demo therefore uses the browser, which carries the same phone-quality
> (300-3400 Hz, narrowband) stimulus and the same scoring.

```mermaid
sequenceDiagram
    autonumber
    participant O as Operator (loopback :8001)
    participant S as Sevi IVR
    participant L as Ledger
    participant T as Twilio
    participant C as Caller's phone

    O->>S: dial {participant_id, lang}
    S->>S: verified caller ID? one call at a time?
    S->>L: check_dial (trial-minutes stop)
    S->>T: create_call (time_limit up to 600 s) (I01)
    T->>C: ring
    C-->>T: answer
    T->>S: POST /twilio/voice/{sid} (signature + call token) (I03)
    S-->>T: TwiML consent: Play + Gather dtmf
    loop each presentation
        T->>S: POST /twilio/step/{sid}/{step} (Digits)
        S->>S: score with the same AnswerScorer, next SNR
        S-->>T: TwiML: Play /audio/{t}/{name}.wav + Gather
        T->>S: GET /audio/{t}/{name}.wav (same mu-law bytes as the browser)
    end
    S->>S: classify (NO_BAND unless a referral rule matches)
    S-->>T: TwiML result wording, Hangup
    T->>S: POST /twilio/status/{sid}
    S->>L: trial minutes for this CallSid
```

- **Every webhook**: Twilio signature (rebuilt from the stored base URL) -> call token
  (constant-time sha256 compare) -> cached TwiML for `(session, step)`, so a Twilio retry gets
  identical TwiML and changes nothing.
- **Optional `<Record>` + Prisma REST** only with `SEVI_TWILIO__RECORD_ANSWERS=true`; the recording
  is deleted right after transcription, whatever happens.
- Outbound only, only to verified caller IDs. The phone record moves `DIALING -> CONSENT ->
  INSTRUCTIONS -> TEST -> ENDED`, driven by webhooks (rules `I01..I15`) instead of the browser FSM.

---

## 13 · Privacy and data

From the consent screen (`prompt.consent_public` in `src/sevi/copy/copy.yaml`):

- In voice mode, **Gnani** processes spoken answers to turn them into text.
- **Tailscale** relays the encrypted connection and cannot read it.
- **Cloudflare** carries the connection only while the backup link is on.
- **GitHub** hosts the page.
- Sevi deletes its copy of the answers after **72 hours** (`src/sevi/app/retention.py`; logs too).

Logs and telemetry are redacted before they are written (`src/sevi/app/logging.py`,
`src/sevi/app/telemetry.py`). Evon sees only the band and the pre-written wording: no transcripts,
digits, scores or names.

```mermaid
flowchart LR
    U["Listener"] -->|"mic audio, voice mode only"| SV["Sevi server"]
    SV -->|"audio frames"| G["Gnani Prisma<br/>speech to text"]
    SV -->|"band + pre-written wording"| EV["Evon"]
    SV --> DB[("SQLite store<br/>sessions, trials, ledger,<br/>ip_hits as HMAC")]
    SV --> LOG[("hourly JSONL logs<br/>redacted")]
    SV -->|"report link r.html#token"| U
    DB -.->|"only the token's sha256 is stored"| TOK["report token"]
    RET["retention.py, every job interval"] -->|"delete after 72 h<br/>(STUDY: strip identifiers)"| DB
    RET -->|"delete after 72 h"| LOG
    RPT["GET /api/v1/report<br/>Bearer token"] -->|"410 after 72 h or unknown"| DB
```

- **Report token.** A 22-character random token is shown once; the server keeps only its sha256.
  The link puts it in the URL fragment (`r.html#...`), which browsers do not send to servers; the
  page sends it as a Bearer header. Deleted and never-existed reports are indistinguishable (410).
- **IP addresses** are kept only as an HMAC, and only when a trusted ingress supplies a client IP.
- **Retention also** recovers orphaned sessions at startup (ended with the T8 decision), settles
  stale ledger reservations, drops idempotency rows and IP hits after 24 h, and compacts the
  database after deletes.

---

## 14 · Results

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
  now 800 ms (#94). The answer settle window (how long Sevi waits after a transcript for more
  digits) is max(`t_settle_ms`, `min_silence` + 300 ms), so 1,100 ms by default: an answer
  with a short mid-answer pause ("five three ... one") is still scored as one triplet.
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

---

## 15 · Limitations

- **Screening, not a diagnosis.** Sevi does not measure hearing thresholds per ear or frequency;
  a REFER or BORDERLINE result points to a professional check, and a PASS is not a clean bill of
  health.
- **Bands need norms.** A band needs a normal-hearing norm cell with n >= 6 on the same frozen
  bank; without one the result is NO_BAND. English is never banded, and phone results are
  NO_BAND unless a referral rule matches.
- **The listener's device is uncalibrated.** Level is controlled relative to the noise (SNR), not
  in absolute dB SPL; speaker playback is not comparable with norms (`SPEAKER_TRANSDUCER`).
- **Voice capture depends on Prisma's VAD.** Below 0 dB SNR the benchmark measured VAD misses,
  not recognition; the keypad path does not have this dependency.
- **Small n.** The Prisma benchmark has 6 triplets per cell, and the Evon pass rate is 19 cases.
- **Known scoring gaps**: extra homophones and Hindi digit words (see Results); both fail safe.

---

## 16 · Deployment

```mermaid
flowchart LR
    subgraph Laptop["Laptop: make serve"]
        PUB["public listener<br/>127.0.0.1:8000"]
        OPL["operator listener<br/>127.0.0.1:8001"]
        DB[("SQLite + var/run/*.env")]
    end
    INF["Infisical<br/>infisical run -- make serve"] -.->|"secrets as env"| PUB
    TS["Tailscale Funnel<br/>https://host.ts.net"] -->|"make ingress"| PUB
    CF["Cloudflare quick tunnel<br/>backup, MODE=quick"] -.-> PUB
    GH["GitHub Pages<br/>web/static via scripts/publish_pages.sh"] -->|"page calls API + WS"| TS
    USER["Listener's browser"] --> GH
    USER --> TS
    subgraph Kaggle["Kaggle 2x T4 session: make evon-up"]
        LLAMA["llama-server (Evon)"]
    end
    LLAMA -->|"trycloudflare URL to var/run/evon.env"| PUB
    PUB --> GN["Gnani Prisma + Timbre"]
    PUB --> TWL["Twilio (phone demo)"]
    OPL -.->|"never exposed through ingress"| X["Tailscale / loopback only"]
```

- **One public entry point.** `scripts/ingress_up.sh` tears down the other mode before starting
  one (Funnel by default, Cloudflare quick tunnel as backup) and writes `var/run/ingress.env`.
- **Evon on Kaggle** stops itself after `EVON_MAX_HOURS` (default 2) or `EVON_IDLE_MIN` (default
  30) idle minutes; the session key is generated per session and never printed.
- **Demo day, one command.** `make demo-up` starts Evon on Kaggle (about 6 minutes), the Funnel and
  `sevi serve` with Infisical secrets, sets `SEVI_PAGES_ORIGIN` to the public URL, and waits for
  readiness `L0`; `make demo-status` prints admission, Evon and credit headroom; `make demo-down`
  stops everything. `DEMO_BUDGET` (default 2500 credits per key) and `DEMO_STOP_SESSIONS`
  (default 3 voice sessions held back on the public key) bound the spend. Open
  `https://<host>.ts.net/sevi/` on any device.
- **Live verification (2026-11-08).** Through the public Funnel URL: Tamil and English keypad and
  voice sessions completed 24/24 with result, Evon explanation and the 24-row report.
- **GitHub Pages on a private repo** needs a paid plan; without it, the app serves `web/static` at
  `<SEVI_PUBLIC_BASE_URL>/sevi/` when `SEVI_ENV != prod` (see `ops/ingress.md`).

---

## 17 · Getting started

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

Other targets: `make e2e` (whole app on uvicorn with fakes and a scripted browser, about 90 s),
`make browser-e2e` (Chromium via Playwright, about 8 min), `make check-slow`, `make cov`,
`make web-test` (node tests for the web client), `make pages-build`.

---

## 18 · Project structure

```text
sevi/
├── src/sevi/
│   ├── contracts/      enums, constants, wire models, ports (shared by both halves)
│   ├── core/           staircase, scoring + lexicon, result + wording, bank, stimulus, dsp, guard
│   ├── calibration/    render, noise, freeze, testbank, bench_prisma, scoring_accuracy
│   ├── copy/copy.yaml  every user-facing string, Tamil and English
│   ├── adapters/       prisma_stream, prisma_rest, timbre, evon, twilio_rest, ledger, breaker, key_semaphore
│   ├── transport/      web_ws (browser CallTransport), uplink
│   ├── app/            main, orchestrator, admission, ivr, explain, store, retention, telemetry, routes/, migrations/
│   ├── fakes/          fake Prisma / Evon / Timbre / Twilio, scenarios
│   └── cli.py
├── web/static/         index, test, result, r (report) pages; js/ (session, wire, g711, audio, worklets)
├── web/tests/          node tests for the web client
├── stimuli/            ta/, en/ frozen banks (digits, noise.wav, manifest.json), prompts/, test-bank/
├── scripts/            render_prompts, ingress_up, publish_pages, evon_kaggle, browser_e2e, ...
├── ops/                ingress.md, evon/ (Kaggle kernel)
├── tests/              unit, contract, property, integration, golden, evals, live
└── Makefile
```

---

## 19 · Engineering discipline

- **1604 tests pass** in `make test` (live, slow and e2e excluded); live tests are credit-capped
  and opt-in.
- **ruff** lint and format check on the whole repo; **mypy `--strict`** on contracts, core, app,
  adapters, transport, fakes and the CLI.
- **Pure core.** Staircase, scoring, result classification and the guard are pure functions;
  the orchestrator runs on a fake clock in tests.
- **Spec rule ids in code.** Every FSM transition (`S..`, `P..`, `I..`) and every result rule
  (`T1..T15`) is recorded, so a session's path can be audited against the contracts.
- **Fail closed.** Unset credit rates refuse billed calls; an unknown report token and an expired
  one look the same; an unfrozen or re-hashed bank is refused at startup.

---

## 20 · Who built what

- **Person A (NithishaVenkatesh):** measurement core, calibration, stimulus banks, copy and
  wording, Timbre prompts.
- **Person B (theCodeForgerHQ):** adapters, transport, app, web client, ops.
