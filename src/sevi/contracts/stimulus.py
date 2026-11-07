"""Bank manifest, triplet design, timing map, rendered trial, prompt manifest (contracts §4.4)."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .enums import BankVariant, Condition, Lang, ModelVersion, OffsetSource, PromptKey
from .types import Digit, NonNegMs, SeviModel, Sha256Hex, SnrDb, Triplet, UInt64, UtcDatetime


def _pinned(value: float) -> object:
    """A float fixed to one value (the spec's ``Literal[<float>]``; mypy forbids float literals)."""
    return Field(ge=value, le=value, allow_inf_nan=False)


Lead500 = Annotated[float, _pinned(500.0)]
Gap200 = Annotated[float, _pinned(200.0)]
Ramp20 = Annotated[float, _pinned(20.0)]
Zero = Annotated[float, _pinned(0.0)]
MixRms20 = Annotated[float, _pinned(-20.0)]
Peak3 = Annotated[float, _pinned(-3.0)]


class DigitAsset(SeviModel):
    digit: Digit
    word: str = Field(min_length=1, max_length=32)  # exact native-script text sent to Timbre
    # 16 kHz PCM16 mono, trimmed, RMS-equalised, offset NOT applied
    file: str = Field(pattern=r"^digits/[0-9]\.wav$")
    sha256: Sha256Hex
    n_samples: int = Field(gt=0)  # at 16 kHz
    rms_dbfs: float  # identical for all digits of a bank (+-0.05)
    offset_db: float = Field(ge=-6.0, le=6.0)  # per-digit level correction applied at mix time
    onset_trim_ms: float = Field(ge=0.0, le=50.0)


class TimbreRenderConfig(SeviModel):
    model: Literal["timbre-v3.0", "timbre-v2.5"]
    voice: str = Field(min_length=1, max_length=32)
    language: Literal["ta-IN", "en-IN"]
    speed: float = Field(1.0, ge=0.85, le=1.15)
    sample_rate_hz: Literal[48000] = 48000
    container: Literal["wav"] = "wav"
    encoding: Literal["linear_pcm"] = "linear_pcm"


class ChainSpec(SeviModel):
    mix_sample_rate_hz: Literal[16000] = 16000
    bandpass: Literal["butter4-300-3400-sos-sosfilt"] = "butter4-300-3400-sos-sosfilt"
    resampler: Literal["resample_poly(1,2)"] = "resample_poly(1,2)"
    wire_sample_rate_hz: Literal[8000] = 8000
    codec: Literal["g711-mulaw-itu-table"] = "g711-mulaw-itu-table"
    prisma_sample_rate_hz: Literal[8000, 16000] = 8000  # G2 outcome; part of bank_hash (F24)
    prisma_upsampler: Literal["none", "resample_poly(2,1)"] = "none"


class TimingSpec(SeviModel):
    lead_ms: Lead500 = 500.0
    gap_ms: Gap200 = 200.0
    lag_ms: Lead500 = 500.0
    ramp_ms: Ramp20 = 20.0


class LevelSpec(SeviModel):
    mix_rms_dbfs: MixRms20 = -20.0
    peak_max_dbfs: Peak3 = -3.0


class TripletDesign(SeviModel):
    n_digits: Literal[8, 9]
    offset_pairs: tuple[tuple[int, int], ...]  # == constants.OFFSET_PAIRS[n_digits]
    pool_size: Literal[120, 126]
    calibration_subset_size: Literal[54, 56]


class CalibrationProvenance(SeviModel):
    source: OffsetSource
    iterations: int = Field(ge=0, le=3)
    prisma_model: Literal["prisma-v3.0", "prisma-v2.5"] | None
    prisma_format: Literal["transcribe", "codemix", "verbatim"] | None
    keyterms_sha256: Sha256Hex | None  # sha of canonical JSON keyterm list, None if off
    prisma_sample_rate_hz: Literal[8000, 16000] | None
    listener_fit_sha256: Sha256Hex | None  # sha256 of results/listener_fit.json at freeze
    panel_outcome_sha256: Sha256Hex | None


class BankManifest(SeviModel):
    schema_id: Literal["sevi.bank/1"] = "sevi.bank/1"
    lang: Lang
    bank_version: str = Field(pattern=r"^(ta|en)-\d+\.\d+\.\d+$")
    variant: BankVariant
    model_version: ModelVersion
    timbre: TimbreRenderConfig
    # ascending; ta == (1..8), en == (1,2,3,4,5,6,8,9) (§4.13; contracts.languages.DIGIT_SETS)
    digit_set: tuple[Digit, ...] = Field(min_length=8, max_length=9)
    digits: tuple[DigitAsset, ...]
    noise_file: Literal["noise.wav"] = "noise.wav"  # 60 s, 16 kHz PCM16, LTASS of corrected digits
    noise_sha256: Sha256Hex
    noise_seed: UInt64
    design: TripletDesign
    timing: TimingSpec = TimingSpec()
    level: LevelSpec = LevelSpec()
    chain: ChainSpec = ChainSpec()
    calibration: CalibrationProvenance
    experimental: bool  # True for en and test-bank
    created_at: UtcDatetime
    frozen: bool = False
    frozen_at: UtcDatetime | None = None
    bank_hash: Sha256Hex
    # bank_hash = sha256(canonical_json(
    #     self.model_dump(mode="json", exclude={"bank_hash", "frozen", "frozen_at"})))

    @model_validator(mode="after")
    def _check(self) -> "BankManifest":
        if [d.digit for d in self.digits] != list(self.digit_set):
            raise ValueError("digits must match digit_set order")
        if self.design.n_digits != len(self.digit_set):
            raise ValueError("design.n_digits != len(digit_set)")
        return self


class DigitSpan(SeviModel):
    position: Literal[1, 2, 3]
    digit: Digit
    onset_ms: NonNegMs  # from clip start, on the 8 kHz output timeline
    offset_ms: NonNegMs


class TimingMap(SeviModel):
    clip_ms: NonNegMs
    noise_on_ms: Zero = 0.0
    noise_off_ms: NonNegMs
    digits: tuple[DigitSpan, DigitSpan, DigitSpan]
    last_digit_offset_ms: NonNegMs  # == digits[2].offset_ms


class RenderedTrial(SeviModel):
    presentation_id: int = Field(ge=1)
    lang: Lang
    triplet: Triplet
    snr_db: SnrDb
    condition: Condition
    noise_seed: UInt64
    noise_start_sample: int = Field(ge=0)  # at 16 kHz
    bank_hash: Sha256Hex
    # G.711 mu-law 8 kHz mono, no header. Plain bytes (R-01): never Base64Bytes;
    # JSON (de)serialisation is base64 via the base model's ser/val_json_bytes
    ulaw: bytes = Field(min_length=1)
    n_bytes: int = Field(gt=0)  # == len(ulaw) == round(clip_ms * 8)
    timing: TimingMap
    level_limited: bool  # True if peak rule reduced the level


class PromptAsset(SeviModel):
    key: PromptKey
    lang: Lang
    copy_key: str  # key in src/sevi/copy/copy.yaml whose text was rendered
    text_sha256: Sha256Hex
    file: str = Field(pattern=r"^[A-Z0-9_]+\.ulaw$")
    sha256: Sha256Hex
    duration_ms: NonNegMs


class PromptManifest(SeviModel):
    schema_id: Literal["sevi.prompts/1"] = "sevi.prompts/1"
    lang: Lang
    voice: str  # "placeholder" for the --placeholder set; prod rejects it
    timbre_model: Literal["timbre-v3.0", "timbre-v2.5"]
    copy_sha256: Sha256Hex  # sha256 of src/sevi/copy/copy.yaml at render time
    assets: tuple[PromptAsset, ...]
    prompts_hash: Sha256Hex
