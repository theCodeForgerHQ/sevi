"""Prisma streaming adapter, pure parts and pacer (B spec §5.3, §13.4 U-PRS-*)."""

from __future__ import annotations

import asyncio
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from adapters_helpers import KEYTERMS_TA, RID, FakeWs, stt_cfg

from sevi.adapters.prisma_stream import (
    HEADER_BUDGET,
    STREAM_RESERVE_S,
    TOP_UP_MARGIN_S,
    OffsetMap,
    PrismaOptions,
    PrismaStreamClient,
    PrismaStreamSession,
    Upsampler2x,
    bind_transcript,
    build_headers,
    map_server_message,
    pace_ms,
    prisma_language_code,
    stream_language_code,
    validate_connected,
    wire_rate,
)
from sevi.contracts.adapters import (
    FrameTag,
    SttClosed,
    SttCommitSent,
    SttConnected,
    SttErrorMsg,
    SttProcessing,
    SttSessionConfig,
    SttSpeechStart,
    SttTranscript,
)
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.contracts.errors import BudgetExceeded, ErrorCode, SttError
from sevi.contracts.telemetry import EventName
from sevi.fakes.clock import FakeClock
from sevi.fakes.ledger_kit import open_ledger
from sevi.fakes.null_telemetry import NullTelemetry, RecordingTelemetry
from sevi.fakes.recording_deps import RecordingBreakers

KT = json.dumps(list(KEYTERMS_TA), ensure_ascii=True, separators=(",", ":"))
FRAME = b"\x01\x00" * 512


def test_U_PRS_01_headers_v3_both_golden() -> None:
    h = build_headers(stt_cfg(), PrismaOptions(), "KEY_B")
    assert h == [
        ("x-api-key-id", "KEY_B"),
        ("language_code", "ta-IN"),
        ("lang_code", "ta-IN"),
        ("model", "prisma-v3.0"),
        ("x-model", "prisma-v3.0"),
        ("format", "transcribe"),
        ("x-format", "transcribe"),
        ("x-enable-itn", "true"),
        ("keyterms", KT),
        ("x-keyterms", KT),
        ("x-keywords", KT),
        ("x-sample-rate", "16000"),  # v3 always streams at 16 kHz (live #64)
        ("x-vad-threshold", "0.7"),
        ("x-min-silence-ms", "800"),
        ("x-min-speech-ms", "100"),
        ("x-min-segment-ms", "100"),
        ("x-idle-timeout-sec", "900"),
        ("x-api-request-id", RID),
    ]


@pytest.mark.parametrize(
    ("mode", "mv", "names"),
    [
        ("doc", ModelVersion.V3, ["language_code", "model", "format", "keyterms", "x-keyterms"]),
        ("sdk", ModelVersion.V3, ["lang_code", "x-model", "x-format", "x-enable-itn", "x-keywords"]),  # noqa: E501
        (
            "both",
            ModelVersion.V2_5,
            ["language_code", "lang_code", "model", "x-model", "format", "x-format",
             "x-enable-itn", "bias_list", "bias_score", "itn_native_numerals"],
        ),
    ],
)  # fmt: skip
def test_U_PRS_01_headers_modes(mode: Any, mv: ModelVersion, names: list[str]) -> None:
    h = build_headers(stt_cfg(model_version=mv), PrismaOptions(header_mode=mode), "K")
    assert [n for n, _ in h][1:-7] == names
    d = dict(h)
    if mv is ModelVersion.V2_5:
        assert d["language_code"] == "ta-IN"
        assert d["model"] == "prisma-v2.5"
        assert d["bias_list"] == "onru,one,irandu,two"


def test_U_PRS_02_ascii_and_header_budget() -> None:
    h = dict(build_headers(stt_cfg(), PrismaOptions(), "K"))
    assert h["keyterms"].isascii()
    assert "\\u0b92" in h["keyterms"]  # ஒ escaped
    many = tuple(f"word{i:03d}" for i in range(400))  # ~12 bytes each x 3 headers > 7,000
    h2 = build_headers(stt_cfg(keyterms=many), PrismaOptions(), "K")
    names = [n for n, _ in h2]
    assert "x-keywords" not in names  # dropped first
    assert "keyterms" in names
    assert sum(len(n) + len(v) + 4 for n, v in h2) <= HEADER_BUDGET
    huge = tuple(f"longkeyterm{i:04d}" for i in range(1000))
    h3 = build_headers(stt_cfg(keyterms=huge), PrismaOptions(), "K")
    assert [n for n, _ in h3].count("keyterms") == 1
    assert "x-keyterms" not in [n for n, _ in h3]
    kept = json.loads(dict(h3)["keyterms"])
    assert 0 < len(kept) < 1000
    assert kept == list(huge[: len(kept)])
    assert sum(len(n) + len(v) + 4 for n, v in h3) <= HEADER_BUDGET


def test_U_PRS_03_validate_connected() -> None:
    cfg = stt_cfg()
    ok = {"sample_rate": 16000, "format": "transcribe", "language_code": "ta-IN"}
    assert validate_connected(cfg, None) == "unverified"
    assert validate_connected(cfg, ok) == "ok"
    for bad in ({**ok, "sample_rate": 8000}, {**ok, "format": "verbatim"}):
        with pytest.raises(SttError) as e:
            validate_connected(cfg, bad)
        assert e.value.code is ErrorCode.STT_CONFIG_MISMATCH
    cands = {"sample_rate": 16000, "format": "transcribe", "language_candidates": ["ta"]}
    assert validate_connected(cfg, cands) == "ok"
    # v2.5 echoes only sample_rate + chunk_size (docs): format/language are not checked (#64)
    v25 = stt_cfg(model_version=ModelVersion.V2_5)
    assert validate_connected(v25, {"sample_rate": 8000, "chunk_size": 512}) == "ok"
    assert validate_connected(v25, {**ok, "sample_rate": 8000}) == "ok"
    for bad25 in (
        {"sample_rate": 16000, "chunk_size": 512},
        {"sample_rate": 8000, "chunk_size": 256},
    ):
        with pytest.raises(SttError):
            validate_connected(v25, bad25)
    en = stt_cfg(lang=Lang.EN)
    assert validate_connected(en, {**ok, "language_code": "en-IN"}) == "ok"
    assert dict(build_headers(en, PrismaOptions(), "K"))["language_code"] == "en-IN"


def test_U_PRS_04_map_server_messages() -> None:
    tel = RecordingTelemetry()
    m = map_server_message(
        '{"type":"connected","request_id":"r1","config":{"sample_rate":8000}}', 1.0
    )
    assert m == SttConnected(request_id="r1", config_echo={"sample_rate": 8000})
    assert map_server_message('{"type":"speech_start","utterance_id":"u-3"}', 1) == (
        SttSpeechStart(utterance_id="u-3")
    )
    assert map_server_message('{"type":"processing"}', 1) == SttProcessing()
    raw = (
        '{"type":"transcript","text":"472","audio_duration_ms":980,"segment_id":"s-1",'
        '"segment_index":1,"latency":318,"is_final":true}'
    )
    assert map_server_message(raw, 5.0) == SttTranscript(
        text="472",
        segment_id="s-1",
        segment_index=1,
        audio_duration_ms=980,
        latency_ms=318,
        detected_language_code=None,
        received_mono_ms=5.0,
    )
    e = map_server_message('{"type":"error","message":"STT engine failed to initialize"}', 1)
    assert e == SttErrorMsg(message="STT engine failed to initialize")
    assert map_server_message('{"type":"transcript","segment_index":2}', 1, tel) is None
    assert tel.of(EventName.STT_ERROR) == [{"code": "STT_SCHEMA"}]
    assert map_server_message('{"type":"vad_end"}', 1, tel) is None
    assert map_server_message("{garbage", 1, tel) is None
    assert len(tel.of(EventName.WIRE_UNKNOWN_MESSAGE)) == 2
    t = map_server_message('{"type":"transcript","text":"","latency":"x"}', 1)
    assert isinstance(t, SttTranscript)
    assert t.latency_ms is None


def _tag(pid: int) -> FrameTag:
    return FrameTag(presentation_id=pid, window_index=1, replay=0)


def test_U_PRS_30_binding_arithmetic() -> None:
    om = OffsetMap()
    for t in range(0, 6400, 64):
        om.append(float(t), _tag(5))
    for t in range(6400, 9600, 64):
        om.append(float(t), None)
    for t in range(9600, 12800, 64):
        om.append(float(t), _tag(6))
    assert om.starts == [0.0, 6400.0, 9600.0]
    tr = SttTranscript(
        text="472",
        segment_id=None,
        segment_index=1,
        audio_duration_ms=980,
        latency_ms=310,
        detected_language_code=None,
        received_mono_ms=0,
    )
    b = bind_transcript(tr, om, 7000.0, 300)
    assert b.stream_mid_ms == 5900.0
    assert b.bound == _tag(5)
    b2 = bind_transcript(tr, om, 11000.0, 300)
    assert b2.stream_mid_ms == 9900.0
    assert b2.bound == _tag(6)
    nolat = tr.model_copy(update={"latency_ms": None})
    assert bind_transcript(nolat, om, 7000.0, 300).stream_mid_ms == 7000 - 400 - 300 - 490
    assert bind_transcript(tr, om, 9000.0, 300).bound is None  # mid 7,900 in the None run
    assert bind_transcript(tr, om, 500.0, 300).bound is None  # mid < 0


def _session(
    clock: FakeClock, ws: FakeWs, cfg: SttSessionConfig | None = None, **kw: Any
) -> PrismaStreamSession:
    """Default: v2.5 at 8 kHz (no resampling, 64 ms frames, 56 ms pace)."""
    return PrismaStreamSession(
        ws,
        cfg or stt_cfg(model_version=ModelVersion.V2_5),
        request_id="r",
        config_echo={},
        clock=clock,
        telemetry=NullTelemetry(),
        **kw,
    )


@pytest.mark.asyncio
async def test_U_PRS_05_pacer_56ms_spacing_and_runs() -> None:
    clock = FakeClock()
    ws = FakeWs(clock)
    s = _session(clock, ws)
    for i in range(10):
        s.send_frame(FRAME, _tag(1) if i < 6 else _tag(2))
    await clock.run_until_idle(200)
    times = [t for t, d in ws.sent if isinstance(d, bytes)]
    assert len(times) == 10
    assert [b - a for a, b in itertools.pairwise(times)] == [56.0] * 9
    assert s.offsets.starts == [0.0, 384.0]
    assert s.offsets.tags == [_tag(1), _tag(2)]
    assert s.written_ms == 640.0
    await s.close()


@pytest.mark.asyncio
async def test_U_PRS_06_frame_length_and_closed_and_sync() -> None:
    clock = FakeClock()
    s = _session(clock, FakeWs(clock))
    with pytest.raises(ValueError, match="1024"):
        s.send_frame(b"x" * 1000, None)
    assert s.send_frame(FRAME, None) is None  # synchronous, not a coroutine
    assert s.commit() is None
    await s.close()
    with pytest.raises(SttError) as e:
        s.send_frame(FRAME, None)
    assert e.value.code is ErrorCode.STT_DISCONNECTED
    with pytest.raises(SttError):
        s.commit()
    await s.close()  # idempotent


@pytest.mark.asyncio
async def test_U_PRS_31_commit_after_queued_frames_and_send_timeout() -> None:
    clock = FakeClock()
    ws = FakeWs(clock)
    s = _session(clock, ws)
    for _ in range(4):
        s.send_frame(FRAME, _tag(1))
    s.commit()  # returns at once
    events: list[Any] = []

    async def consume() -> None:
        async for ev in s.events():
            events.append(ev)
            if isinstance(ev, SttCommitSent):
                return

    await asyncio.wait_for(consume(), 2)
    kinds = ["audio" if isinstance(d, bytes) else d for _, d in ws.sent]
    assert kinds == ["audio"] * 4 + ['{"type":"commit"}']
    assert events[-1].t_mono_ms == ws.sent[3][0]  # right after the 4th frame (3 x 56 ms in)
    assert ws.sent[3][0] - ws.sent[0][0] == 3 * 56.0
    await s.close()

    stalled = FakeWs(clock, stall=True)
    s2 = _session(clock, stalled, send_timeout_s=0.05)
    s2.send_frame(FRAME, None)
    s2.commit()
    got = [ev async for ev in s2.events()]
    assert got[-1] == SttClosed(code=None, reason="send_timeout")
    await s2.close()


@pytest.mark.asyncio
async def test_gated_zero_frames_dropped_only_under_backlog() -> None:
    clock = FakeClock()
    ws = FakeWs(clock)
    s = _session(clock, ws)
    for _ in range(12):
        s.send_frame(bytes(1024), None, gated_zero=True)
    s.send_frame(FRAME, _tag(1))  # speech is never dropped
    await clock.run_until_idle(200)
    sent = [d for _, d in ws.sent]
    assert FRAME in sent
    assert s.dropped_gated == 5  # dropped while > 8 frames (incl. itself) were waiting
    await s.close()


@pytest.mark.asyncio
async def test_U_PRS_08_ledger_bills_max_frames_or_wall_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    kit = await open_ledger(tmp_path / "l.sqlite", budget=1000.0, clock=clock)
    r = await kit.ledger.reserve(KeyId.B, "public_voice", stt_s=STREAM_RESERVE_S)
    s = _session(clock, FakeWs(clock), ledger=kit.ledger, reservation=r, reserved_s=60.0)
    for _ in range(1000):  # 64 s of audio, paced at 56 ms -> 55.9 s of FakeClock
        s.send_frame(FRAME, None)
    await clock.run_until_idle(5000)
    assert s.frames_written == 1000
    assert (await kit.reservations())[0]["credits"] == 120.0  # topped up once, early
    clock.advance((s.opened_mono_ms + 70_000.0 - clock.mono_ms()) / 1000.0)  # 70 s wall
    await s.close()
    events = await kit.events()
    assert [e.stt_s for e in events] == [70.0]
    assert events[0].purpose == "public_voice"
    assert events[0].bucket == "PUBLIC"
    assert await kit.reservations() == []
    await kit.close()


@pytest.mark.asyncio
async def test_close_cancelled_by_caller_timeout_still_commits_ledger(tmp_path: Path) -> None:
    """Live finding (2026-11-08): the orchestrator bounds close() with wait_for(1 s); a slow
    socket close cancelled it before the commit and left the reservation open."""

    class HangingCloseWs(FakeWs):
        async def close(self, code: int = 1000, reason: str = "") -> None:
            await asyncio.Event().wait()

    clock = FakeClock()
    kit = await open_ledger(tmp_path / "l.sqlite", budget=1000.0, clock=clock)
    r = await kit.ledger.reserve(KeyId.B, "public_voice", stt_s=STREAM_RESERVE_S)
    s = _session(clock, HangingCloseWs(clock), ledger=kit.ledger, reservation=r, reserved_s=60.0)
    s.send_frame(FRAME, None)
    await clock.run_until_idle(200)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(s.close(), 0.05)
    for _ in range(50):
        if not await kit.reservations():
            break
        await asyncio.sleep(0.01)
    assert await kit.reservations() == []
    assert len(await kit.events()) == 1
    await kit.close()


@pytest.mark.asyncio
async def test_stream_top_up_early_and_budget_stop_sends_nothing_unreserved(
    tmp_path: Path,
) -> None:
    """Budget = exactly one reservation (60 credits at 1 credit/s): the top-up is refused while
    reserved time is still left, the stream ends with reason=budget, the socket is closed, no
    frame is sent after the refusal and the committed seconds stay within the reservation."""
    clock = FakeClock()
    kit = await open_ledger(tmp_path / "l.sqlite", budget=60.0, clock=clock)
    r = await kit.ledger.reserve(KeyId.B, "public_voice", stt_s=STREAM_RESERVE_S)
    ws = FakeWs(clock)
    s = _session(clock, ws, ledger=kit.ledger, reservation=r, reserved_s=60.0)
    for _ in range(1200):  # 76.8 s of audio wanted
        s.send_frame(FRAME, None)
    got = [ev async for ev in s.events()]
    assert got[-1] == SttClosed(code=None, reason="budget")
    assert ws.closed
    sent = sum(1 for _, d in ws.sent if isinstance(d, bytes))
    assert sent == s.frames_written
    assert sent * 0.064 <= STREAM_RESERVE_S - TOP_UP_MARGIN_S + 0.064
    with pytest.raises(SttError):
        s.send_frame(FRAME, None)
    await clock.run_until_idle(50)
    assert sum(1 for _, d in ws.sent if isinstance(d, bytes)) == sent
    await s.close()
    events = await kit.events()
    assert len(events) == 1
    assert events[0].stt_s <= STREAM_RESERVE_S
    assert await kit.reservations() == []
    await kit.close()


class _InvalidStatusConnect:
    """connect() stand-in: raise InvalidStatus with scripted statuses, then a FakeWs."""

    def __init__(self, statuses: list[int], clock: FakeClock) -> None:
        self.statuses = statuses
        self.clock = clock
        self.calls = 0

    async def __call__(self, url: str, **kw: Any) -> FakeWs:
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response

        self.calls += 1
        if self.statuses:
            st = self.statuses.pop(0)
            raise InvalidStatus(Response(st, "x", Headers()))
        ws = FakeWs(self.clock)
        ws.incoming.put_nowait(
            json.dumps(
                {
                    "type": "connected",
                    "request_id": "r1",
                    "config": {
                        "sample_rate": 16000,
                        "format": "transcribe",
                        "language_code": "ta-IN",
                    },
                }
            )
        )
        return ws


@pytest.mark.asyncio
async def test_U_PRS_07_open_retries_two_429_then_success(tmp_path: Path) -> None:
    clock = FakeClock()
    conn = _InvalidStatusConnect([429, 429], clock)
    breakers = RecordingBreakers()
    kit = await open_ledger(tmp_path / "l.sqlite", clock=clock)
    client = PrismaStreamClient(
        PrismaOptions(),
        {KeyId.B: "K"},
        breakers,
        kit.ledger,
        clock,
        NullTelemetry(),
        connect=conn,
        rng=random.Random(7),  # noqa: S311
    )
    s = await client.open(stt_cfg())
    assert conn.calls == 3
    exp = random.Random(7)  # noqa: S311
    d0, d1 = exp.uniform(0, 0.5), exp.uniform(0, 1.0)
    assert clock.sleeps[:2] == pytest.approx([d0 * 1000, d1 * 1000])
    assert breakers.get("prisma", KeyId.B).failures == 0
    await s.close()
    await kit.close()


@pytest.mark.asyncio
async def test_open_refused_by_budget_before_any_network_io(tmp_path: Path) -> None:
    clock = FakeClock()
    conn = _InvalidStatusConnect([], clock)
    kit = await open_ledger(tmp_path / "l.sqlite", budget=10.0, clock=clock)  # needs 60
    ledger = kit.ledger
    client = PrismaStreamClient(
        PrismaOptions(), {KeyId.B: "K"}, RecordingBreakers(), ledger, clock, NullTelemetry(),
        connect=conn,
    )  # fmt: skip
    with pytest.raises(BudgetExceeded):
        await client.open(stt_cfg())
    assert conn.calls == 0
    assert await kit.events() == []
    assert await kit.reservations() == []
    await kit.close()


def test_frame_ms_matches_documented_chunking() -> None:
    """docs: 512 samples / 1,024 bytes per frame -> 64 ms at 8 kHz, 32 ms at 16 kHz (#64)."""
    from sevi.adapters.prisma_stream import frame_ms

    assert frame_ms(8000) == 64.0
    assert frame_ms(16000) == 32.0


def test_stream_language_code_is_one_locale_rest_unchanged() -> None:
    """Live #64: the v3 stream rejects "ta-IN,en-IN"; REST keeps the C§4.13 codes."""
    for mv in ModelVersion:
        assert stream_language_code(Lang.TA, mv) == "ta-IN"
        assert stream_language_code(Lang.EN, mv) == "en-IN"
        for mode in ("doc", "both"):
            h = dict(build_headers(stt_cfg(model_version=mv), PrismaOptions(header_mode=mode), "K"))
            assert h["language_code"] == "ta-IN"
    assert prisma_language_code(Lang.TA, ModelVersion.V3) == "ta-IN,en-IN"


def test_wire_rate_and_pace() -> None:
    """Live #64: v3 always runs at 16 kHz; v2.5 streams the input rate. Pace = 0.875 frame."""
    assert wire_rate(ModelVersion.V3, 8000) == 16000
    assert wire_rate(ModelVersion.V3, 16000) == 16000
    assert wire_rate(ModelVersion.V2_5, 8000) == 8000
    assert pace_ms(8000) == 56.0
    assert pace_ms(16000) == 28.0
    h = dict(build_headers(stt_cfg(model_version=ModelVersion.V2_5), PrismaOptions(), "K"))
    assert h["x-sample-rate"] == "8000"


def test_upsampler_continuous_across_frame_boundaries() -> None:
    """A 440 Hz sine upsampled frame by frame equals the one-shot result and is a clean
    16 kHz sine (no clicks at the 512-sample frame edges)."""
    n = np.arange(512 * 6)
    x = (8000 * np.sin(2 * np.pi * 440 * n / 8000)).astype("<i2")
    up = Upsampler2x()
    y = np.frombuffer(
        b"".join(up.process(x[i : i + 512].tobytes()) for i in range(0, x.size, 512)), "<i2"
    ).astype(np.float64)
    one = np.frombuffer(Upsampler2x().process(x.tobytes()), "<i2").astype(np.float64)
    assert y.size == 2 * x.size
    assert np.array_equal(y, one)
    delay = (Upsampler2x.TAPS - 1) // 2
    m = np.arange(y.size)
    ref = 8000 * np.sin(2 * np.pi * 440 * (m - delay) / 16000)
    steady = slice(Upsampler2x.TAPS, None)
    assert np.max(np.abs(y[steady] - ref[steady])) < 80  # < 1 % of amplitude
    jumps = np.abs(np.diff(y[steady]))
    assert jumps.max() < 2 * np.pi * 440 / 16000 * 8000 * 1.1  # max slope of the sine


@pytest.mark.asyncio
async def test_v3_session_upsamples_paces_28ms_and_bills_16k_frames() -> None:
    clock = FakeClock()
    ws = FakeWs(clock)
    s = _session(clock, ws, stt_cfg())
    for i in range(4):
        s.send_frame(FRAME, _tag(1) if i < 2 else _tag(2))
    await clock.run_until_idle(200)
    sent = [(t, d) for t, d in ws.sent if isinstance(d, bytes)]
    assert len(sent) == 8
    assert all(len(d) == 1024 for _, d in sent)
    assert [b[0] - a[0] for a, b in itertools.pairwise(sent)] == [28.0] * 7
    assert s.frames_written == 8
    assert s.written_ms == 256.0
    assert s.offsets.starts == [0.0, 128.0]
    assert s.billed_s() == pytest.approx(
        max(8 * 0.032, (clock.mono_ms() - s.opened_mono_ms) / 1000)
    )
    await s.close()
