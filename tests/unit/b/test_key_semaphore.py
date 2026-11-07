"""Key semaphore (B spec §5.2): U-SEM-01..05, P-SEM-01, U-CFG-04 (semaphore part)."""

from __future__ import annotations

import asyncio

import pytest
from b_fakes import Clock, RecordingTelemetry
from hypothesis import given, settings
from hypothesis import strategies as st

from sevi.adapters.key_semaphore import ProcessKeySemaphore
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import ErrorCode, SeviError


def _sem(
    a: int = 1, b: int = 1, *, alias: bool = False, coexist: bool = False
) -> tuple[ProcessKeySemaphore, RecordingTelemetry]:
    tel = RecordingTelemetry()
    sem = ProcessKeySemaphore(
        {KeyId.A: a, KeyId.B: b},
        alias_b=alias,
        stream_rest_coexist=coexist,
        clock=Clock(),
        telemetry=tel,
    )
    return sem, tel


@pytest.mark.asyncio
async def test_U_SEM_01_timeout_zero_then_release() -> None:
    sem, _ = _sem()
    p = await sem.take(KeyId.A, "stream", 0)
    with pytest.raises(SeviError) as ei:
        await sem.take(KeyId.A, "stream", 0)
    assert ei.value.code is ErrorCode.KEY_SEMAPHORE_TIMEOUT
    assert sem.try_take_stream(KeyId.A) is None
    p.release()
    assert sem.try_take_stream(KeyId.A) is not None


@pytest.mark.asyncio
async def test_U_SEM_02_tts_never_with_stream() -> None:
    sem, tel = _sem(a=2)
    async with sem.acquire(KeyId.A, "stream", 0):
        with pytest.raises(SeviError):
            await sem.take(KeyId.A, "tts", 0.05)
    assert "semaphore.timeout" in tel.names()
    async with sem.acquire(KeyId.A, "tts", 0):
        assert sem.try_take_stream(KeyId.A) is None
        assert sem.free_stream(KeyId.A) == 0


@pytest.mark.asyncio
async def test_U_SEM_03_rest_with_stream_only_if_coexist() -> None:
    sem, _ = _sem(a=2, coexist=False)
    stream = await sem.take(KeyId.A, "stream", 0)
    waiter = asyncio.create_task(sem.take(KeyId.A, "rest", 1))
    await asyncio.sleep(0.01)
    assert not waiter.done()
    stream.release()
    rest = await waiter
    assert rest.purpose == "rest"
    sem2, _ = _sem(a=2, coexist=True)
    await sem2.take(KeyId.A, "stream", 0)
    assert (await sem2.take(KeyId.A, "rest", 0)).purpose == "rest"


@pytest.mark.asyncio
async def test_U_SEM_04_fifo() -> None:
    sem, _ = _sem()
    held = await sem.take(KeyId.A, "rest", 0)
    order: list[str] = []

    async def want(tag: str) -> None:
        p = await sem.take(KeyId.A, "rest", 2)
        order.append(tag)
        await asyncio.sleep(0)
        p.release()

    t1 = asyncio.create_task(want("t1"))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(want("t2"))
    await asyncio.sleep(0)
    held.release()
    await asyncio.gather(t1, t2)
    assert order == ["t1", "t2"]


@pytest.mark.asyncio
async def test_U_SEM_05_release_idempotent() -> None:
    sem, _ = _sem(a=2)
    p = await sem.take(KeyId.A, "rest", 0)
    p.release()
    p.release()
    assert sem.pool(KeyId.A).held["rest"] == 0
    assert sem.free_stream(KeyId.A) == 2


@pytest.mark.asyncio
async def test_U_CFG_04_b_pool_is_a_pool_when_aliased() -> None:
    sem, _ = _sem(a=1, b=3, alias=True)
    assert sem.pool(KeyId.B) is sem.pool(KeyId.A)
    assert sem.try_take_stream(KeyId.B) is not None
    assert sem.try_take_stream(KeyId.A) is None


@pytest.mark.asyncio
async def test_timeout_removes_waiter_and_does_not_leak() -> None:
    sem, _ = _sem()
    p = await sem.take(KeyId.A, "stream", 0)
    with pytest.raises(SeviError):
        await sem.take(KeyId.A, "stream", 0.02)
    p.release()
    assert sem.pool(KeyId.A).waiters == type(sem.pool(KeyId.A).waiters)()
    assert sem.free_stream(KeyId.A) == 1


OPS = st.lists(
    st.tuples(
        st.sampled_from(["take", "release"]),
        st.sampled_from([KeyId.A, KeyId.B]),
        st.sampled_from(["stream", "rest", "tts"]),
        st.sampled_from([0.0, 0.001]),
    ),
    max_size=40,
)


@settings(max_examples=60, deadline=None)
@given(ops=OPS, coexist=st.booleans(), pa=st.integers(1, 3), pb=st.integers(1, 3))
def test_P_SEM_01_invariants(
    ops: list[tuple[str, KeyId, str, float]], coexist: bool, pa: int, pb: int
) -> None:
    async def run() -> None:
        sem, _ = _sem(pa, pb, coexist=coexist)
        held: list[object] = []
        tasks: list[asyncio.Task[object]] = []

        def check() -> None:
            for k in (KeyId.A, KeyId.B):
                pool = sem.pool(k)
                assert sum(pool.held.values()) <= pool.permits
                assert not (pool.held["tts"] and pool.held["stream"])
                assert min(pool.held.values(), default=0) >= 0

        for op, key, purpose, timeout in ops:
            if op == "take":

                async def grab(k: KeyId = key, p: str = purpose, t: float = timeout) -> object:
                    try:
                        permit = await sem.take(k, p, t)  # type: ignore[arg-type]
                    except SeviError:
                        return None
                    held.append(permit)
                    return permit

                tasks.append(asyncio.create_task(grab()))
            elif held:
                held.pop(0).release()  # type: ignore[attr-defined]
            await asyncio.sleep(0)
            check()
        await asyncio.gather(*tasks)
        check()
        for p in held:
            p.release()  # type: ignore[attr-defined]
        check()
        assert sem.free_stream(KeyId.A) == pa

    asyncio.run(run())
