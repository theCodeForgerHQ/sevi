"""Pure orchestrator helpers: open_utterance (P-ORC-01, contracts §6.3.1), timers, gate
intervals, window tags, UUIDv7 session ids."""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from sevi.app.orchestrator import Intervals, Timers, WindowTags, new_session_id, open_utterance
from sevi.contracts.adapters import FrameTag
from sevi.contracts.enums import WindowCloseReason
from sevi.contracts.trial import SttSegment
from sevi.contracts.types import UUID7_RE


def seg(text: str, received_ms: float, latency: int | None = 300) -> SttSegment:
    return SttSegment(text=text, received_ms=received_ms, latency_ms=latency, after_commit=False)


def never(_a: float, _b: float) -> bool:
    return False


def always(_a: float, _b: float) -> bool:
    return True


R = WindowCloseReason


def test_P_ORC_01_clean_three_digits_never_commits() -> None:
    """IT-ORC-14: THREE_DIGITS with no speech after the segment -> no commit."""
    assert not open_utterance(R.THREE_DIGITS, True, [seg("4 7 2", 900)], 0, 1000, None, 0, never)


def test_P_ORC_01_rules_a_to_d() -> None:
    assert open_utterance(R.TIMEOUT, True, [], 0, 8000, None, 0, never)  # (a) and (c)
    assert not open_utterance(R.TIMEOUT, False, [], 0, 8000, None, 0, never)
    # (b): speech after t_seg = 1000 - 300 + 50
    seen: list[tuple[float, float]] = []

    def spy(a: float, b: float) -> bool:
        seen.append((a, b))
        return True

    assert open_utterance(R.THREE_DIGITS, False, [seg("4 7 2", 1000)], 0, 1200, None, 0, spy)
    assert seen == [(750.0, 1200.0)]
    # (b) with unknown latency uses 400 ms
    seen.clear()
    open_utterance(R.SETTLE, False, [seg("4", 1000, None)], 100, 2000, None, 0, spy)
    assert seen == [(750.0, 2000.0)]
    # (c): no non-empty transcript but speech
    assert open_utterance(R.SETTLE, True, [seg("", 500)], 0, 2000, None, 0, never)
    # (d): VAD activity after L, or outstanding processing
    assert open_utterance(R.THREE_DIGITS, False, [seg("4 7 2", 900)], 0, 1000, 950, 0, never)
    assert not open_utterance(R.THREE_DIGITS, False, [seg("4 7 2", 900)], 0, 1000, 850, 0, never)
    assert open_utterance(R.THREE_DIGITS, False, [seg("4 7 2", 900)], 0, 1000, None, 1, never)


@given(
    speech=st.booleans(),
    outstanding=st.integers(0, 3),
    vad=st.one_of(st.none(), st.floats(0, 5000)),
)
def test_P_ORC_01_dtmf_complete_never_commits(
    speech: bool, outstanding: int, vad: float | None
) -> None:
    assert not open_utterance(R.DTMF_COMPLETE, speech, [], 0, 5000, vad, outstanding, always)


@given(
    reason=st.sampled_from(list(R)),
    speech=st.booleans(),
    texts=st.lists(st.sampled_from(["", "4", "4 7 2", "hmm"]), max_size=4),
    outstanding=st.integers(0, 2),
)
def test_P_ORC_01_outstanding_or_speech_without_text_commits(
    reason: R, speech: bool, texts: list[str], outstanding: int
) -> None:
    segs = [seg(t, 100.0 * (i + 1)) for i, t in enumerate(texts)]
    got = open_utterance(reason, speech, segs, 0, 9000, None, outstanding, never)
    if reason is R.DTMF_COMPLETE:
        assert not got
    elif outstanding > 0 or (speech and not any(t.strip() for t in texts)):
        assert got


def test_timers_fire_in_deadline_order_and_rearm_replaces() -> None:
    t = Timers()
    t.arm("A", 300)
    t.arm("B", 100)
    t.arm("A", 200)  # replaces
    assert t.next_at() == 100
    assert t.pop_due(50) is None
    first, second = t.pop_due(1000), t.pop_due(1000)
    assert first is not None
    assert first.name == "B"
    assert second is not None
    assert second.name == "A"
    assert t.pop_due(1000) is None
    assert t.next_at() is None
    t.arm("C", 5)
    t.cancel("C")
    assert t.armed("C") is None


def test_gate_intervals_contain_by_capture_time() -> None:
    g = Intervals()
    h = g.open(1000)
    assert g.contains(5000)  # open-ended until fixed
    g.set_end(h, 3950)
    assert g.contains(1000)
    assert g.contains(3950)
    assert not g.contains(3951)
    assert not g.contains(999)


def test_window_tags() -> None:
    w = WindowTags()
    tag = FrameTag(presentation_id=7, window_index=1, replay=0)
    w.open(tag, 4000)
    assert w.tag_at(3999) is None
    assert w.tag_at(9000) == tag
    w.close(6000)
    assert w.tag_at(5999) == tag
    assert w.tag_at(6001) is None


def test_session_id_is_uuid7() -> None:
    ids = {new_session_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(re.match(UUID7_RE, i) for i in ids)
