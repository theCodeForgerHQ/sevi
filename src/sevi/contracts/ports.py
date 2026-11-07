"""The 4 pure A->B ports and the construction factories B's startup calls (contracts §5.1, §5.1.1).

Person A implements these in `sevi.core.*`. A's models are referenced by module name only
(TYPE_CHECKING imports from `sevi.contracts.{stimulus,trial,procedure,result}`), so this module
imports cleanly before A's contracts land.

Scope (BUILD-BRIEF reduced scope), omitted: `StimulusProvider.render_panel_trial` and the
`variants` parameter of `load_bank` (human panel cut; only the SHIPPED variant is loaded),
`load_norms` (norms cut). `PanelService` (§5.1.2) is cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from sevi.contracts.enums import Lang, ModelVersion, Surface
from sevi.contracts.types import SeviModel

if TYPE_CHECKING:  # Person A's modules (contracts §2.1)
    from sevi.contracts.procedure import ProcedureConfig, ProcedureState, SrtResult
    from sevi.contracts.result import ResultDecision, ResultInput
    from sevi.contracts.stimulus import BankManifest, RenderedTrial
    from sevi.contracts.trial import AnswerEvidence, PresentationSpec, ScoringResult


@runtime_checkable
class StimulusProvider(Protocol):
    """Owner A (core/stimulus.py). One instance per language, built by core.bank.load_bank().
    Pure after construction. Thread-safe (no mutable state)."""

    @property
    def manifest(self) -> BankManifest: ...

    def render_trial(self, spec: PresentationSpec) -> RenderedTrial:
        """Mix spec.triplet at spec.snr_db with the noise segment chosen by spec.noise_seed, apply
        spec.condition, run the 16k -> 300-3400 Hz -> 8k -> mu-law chain. Sync, CPU-bound.
        Budget p95 <= 20 ms. Noise segment (R-11): noise_start_sample =
        Generator(PCG64(spec.noise_seed)).integers(0, len(noise_16k) - n_clip_16k + 1), the only
        draw from that generator. Raises ValueError if a triplet digit is not in
        manifest.digit_set. Identical spec -> identical RenderedTrial.ulaw bytes (G-STIM-01)."""
        ...


@runtime_checkable
class AdaptiveProcedure(Protocol):
    """Owner A (core/staircase.py). Pure, functional. Rules: §4.6.1. Budget <= 1 ms per call."""

    def start(self, config: ProcedureConfig) -> ProcedureState:
        """Initial state with `pending` set. Never raises for a valid config."""
        ...

    def record(self, state: ProcedureState, result: ScoringResult) -> ProcedureState:
        """Apply one ACCEPTed result for state.pending. Returns a new state whose `pending` is the
        next presentation, or None with `terminal` set. Raises
        ProcedureProtocolError(PROCEDURE_PROTOCOL) if state.terminal is set, result.action !=
        ACCEPT, or result.presentation_id != state.pending.presentation_id."""
        ...

    def summarize(self, state: ProcedureState) -> SrtResult:
        """Valid for any state (partial sessions included)."""
        ...


@runtime_checkable
class AnswerScorer(Protocol):
    """Owner A (core/scoring.py). Pure. Built by load_lexicons() at startup."""

    def digits_so_far(self, lang: Lang, texts: Sequence[str]) -> int:
        """Digits the normaliser finds in the space-joined texts (capped at 12). Budget <= 2 ms."""
        ...

    def score(self, spec: PresentationSpec, evidence: AnswerEvidence) -> ScoringResult:
        """Rules §4.5.1. Budget <= 5 ms.
        Raises ValueError if evidence.presentation_id != spec.presentation_id."""
        ...

    def intent_so_far(
        self, lang: Lang, texts: Sequence[str]
    ) -> Literal["DONT_KNOW", "REPEAT"] | None:
        """Intent found when digits_so_far(...) == 0, else None. Budget <= 2 ms."""
        ...

    def keyterms(self, lang: Lang, model_version: ModelVersion) -> tuple[str, ...]:
        """Digit words for Prisma: v3 -> <= 200 items; v2.5 -> <= 100 single words, letters only.
        Deterministic order (lexicon file order)."""
        ...


class WordingText(SeviModel):
    wording_class: str
    surface: Surface
    lang: str
    text: str  # slots already filled (PAGE/REPORT may contain numbers; SPOKEN never)
    copy_key: str
    variant_id: str


@runtime_checkable
class ResultClassifier(Protocol):
    """Owner A (core/result.py + core/wording.py). Pure."""

    def classify(self, inp: ResultInput) -> ResultDecision:
        """§4.7.1 truth table, first match wins. Budget <= 5 ms. Never raises for valid input."""
        ...

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> WordingText:
        """Copy variant for f"wording.{decision.wording_class}.{surface.lower()}" with slots
        {SRT}, {N}, {AGES} filled. SPOKEN text has no slots. Raises KeyError only if
        src/sevi/copy/copy.yaml lacks the entry. Callers call it only for
        `surface in decision.surfaces`."""
        ...


# ---------- factories (§5.1.1; owner A; B's startup calls exactly these) ----------


class BankLoader(Protocol):
    """`sevi.core.bank.load_bank`. Verifies every file sha256 and the manifest bank_hash.
    Raises BankError(BANK_INTEGRITY) on any sha mismatch; BankError(BANK_HASH_REJECTED) if
    (not allow_unfrozen) and bank_hash != allowed_hashes[lang] or manifest.frozen is False.
    Budget <= 2 s per language."""

    def __call__(
        self,
        lang: Lang,
        bank_dir: str,
        allowed_hashes: dict[Lang, str],
        allow_unfrozen: bool,
    ) -> StimulusProvider: ...


class LexiconLoader(Protocol):
    """`sevi.core.scoring.load_lexicons`."""

    def __call__(self, lexicon_dir: str | None = None) -> AnswerScorer: ...


class ClassifierBuilder(Protocol):
    """`sevi.core.result.build_classifier`."""

    def __call__(
        self,
        copy_path: str = "src/sevi/copy/copy.yaml",
        selection_path: str | None = None,
    ) -> ResultClassifier: ...


class T8Decider(Protocol):
    """`sevi.core.result.t8_decision`: the exact T8 decision classify() returns for SERVER_ERROR
    (rule T8, band FAILED, W_SYSFAIL, surfaces SPOKEN/PAGE/REPORT/OPERATOR, srt from `srt` if
    any). Pure; never raises."""

    def __call__(self, srt: SrtResult | None) -> ResultDecision: ...


# `sevi.core.staircase.PROCEDURE: Final[AdaptiveProcedure]` is a stateless module singleton.
