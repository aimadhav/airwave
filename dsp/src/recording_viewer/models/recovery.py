"""Provider-independent contracts for separation and speaker-attributed words."""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from recording_viewer.models import TranscriptWord, validate_interval
from recording_viewer.exceptions import RecoveryError


@dataclass(frozen=True)
class RecoveryWindow:
    """Only core times own words; surrounding crop is inference context."""

    id: str
    start: float
    end: float
    crop_start: float
    crop_end: float
    speakers: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_interval(self.start, self.end)
        validate_interval(self.crop_start, self.crop_end)
        if not self.crop_start <= self.start < self.end <= self.crop_end:
            raise RecoveryError("Recovery core must lie inside its context crop")
        if not self.id or len(set(self.speakers)) != len(self.speakers) or len(self.speakers) < 2:
            raise RecoveryError("Recovery windows require distinct speaker IDs")


@dataclass(frozen=True)
class AttributedWord:
    """An actual separated-track recognition result, in session coordinates."""

    speaker: str
    word: TranscriptWord
    window_id: str
    cosine: float


@dataclass(frozen=True)
class WindowReport:
    """Every attempted or skipped overlap has an inspectable decision."""

    window: RecoveryWindow
    status: Literal["accepted", "rejected", "skipped", "failed"]
    reason: str
    metrics: dict[str, float]
    words: tuple[AttributedWord, ...] = ()
    track_speakers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryResult:
    """Immutable audit envelope; failed attempts never replace the baseline."""

    reports: tuple[WindowReport, ...]
    reference_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeparationOutput:
    """Tracks retain input sample rate, length and common gain; no names implied."""

    tracks: tuple[np.ndarray, ...]
    sample_rate: int
    applied_gain: float = 1.0
    raw_peak: float | None = None
