"""Canonical timestamp contracts; no dependency on a model implementation."""

from dataclasses import dataclass
import math
from recording_viewer.exceptions import TranscriptInputError


def validate_interval(start: float, end: float) -> None:
    """Reject invalid intervals at the presentation boundary."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise TranscriptInputError("Intervals must be finite, non-negative and non-empty")


@dataclass(frozen=True)
class TranscriptWord:
    """A recognized word with model-estimated times, not an invented interval."""

    start: float
    end: float
    text: str
    probability: float | None = None

    def __post_init__(self) -> None:
        validate_interval(self.start, self.end)
        if not self.text.strip():
            raise TranscriptInputError("Transcript words cannot be empty")
        if self.probability is not None and (not math.isfinite(self.probability) or not 0 <= self.probability <= 1):
            raise TranscriptInputError("Recognition probability must lie in [0, 1]")


@dataclass(frozen=True)
class VoiceRegion:
    """Anonymous speaker activity, which may overlap another region."""

    id: str
    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        validate_interval(self.start, self.end)
        if not self.id or not self.speaker:
            raise TranscriptInputError("Speaker regions require identifiers")


@dataclass(frozen=True)
class AlignedTurn:
    """Words assigned once; unrecognized secondary voices remain explicit."""

    id: str
    speaker: str
    start: float
    end: float
    text: str
    uncertain: bool
    overlap: bool
    word_count: int
    transcript_source: str = "mixed_asr"
    speaker_source: str = "diarization_timing"


@dataclass(frozen=True)
class TranscriptGroup:
    """A passage of intersecting turns rendered as vertically stacked rows."""

    start: float
    end: float
    turn_ids: tuple[str, ...]
    overlap: bool
