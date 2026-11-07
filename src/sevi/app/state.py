"""`AppState`: the one process-wide state object shared by both listeners (B spec §2.1)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sevi.app.settings import Settings, Switches
from sevi.contracts.adapters import Clock, Telemetry
from sevi.contracts.enums import Lang
from sevi.contracts.ports import AnswerScorer, ResultClassifier, StimulusProvider

if TYPE_CHECKING:
    from sevi.adapters.breaker import BreakerRegistry
    from sevi.adapters.key_semaphore import ProcessKeySemaphore
    from sevi.adapters.ledger import Ledger
    from sevi.app.admission import Admission, CreditGate
    from sevi.app.readiness import Readiness
    from sevi.app.session_port import SessionPort
    from sevi.app.store import SqliteStore
    from sevi.core.wording import WordingTable


@dataclass
class AppState:
    settings: Settings
    switches: Switches
    clock: Clock
    telemetry: Telemetry
    store: SqliteStore
    ledger: Ledger
    credit: CreditGate
    breakers: BreakerRegistry
    sem: ProcessKeySemaphore
    port: SessionPort
    banks: dict[Lang, StimulusProvider]
    scorer: AnswerScorer
    classifier: ResultClassifier
    wording: WordingTable
    git_sha: str
    copy_sha256: str
    prompts_hashes: dict[str, str] = field(default_factory=dict)
    adapters: dict[str, Any] = field(default_factory=dict)
    stub_factories: list[str] = field(default_factory=list)
    draining: bool = False
    startup_retention_done: bool = False
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    # set right after construction (they read the state)
    admission: Admission = field(init=False)
    readiness: Readiness = field(init=False)

    @property
    def languages_unavailable(self) -> list[Lang]:
        return [lang for lang in self.settings.result.enabled_langs if lang not in self.banks]
