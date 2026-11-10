"""Stable data contracts shared by DSP stages and downstream consumers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


SourceType = Literal["google_meet", "audio_file"]
SpeakerSource = Literal["meet_transcript", "diarization", "speaker_match", "unknown"]


@dataclass(frozen=True)
class AudioMetadata:
    """Describes an audio buffer in the canonical DSP representation."""

    source_path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Return the duration represented by the metadata."""

        return self.frame_count / self.sample_rate


@dataclass(frozen=True)
class AudioBuffer:
    """Audio samples plus metadata.

    Samples are shaped ``(frames, channels)`` and use float32 values in the
    range approximately ``[-1, 1]``.
    """

    samples: np.ndarray
    metadata: AudioMetadata

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("Audio samples must have shape (frames, channels)")
        if self.samples.shape[0] != self.metadata.frame_count:
            raise ValueError("Metadata frame count does not match the samples")
        if self.samples.shape[1] != self.metadata.channels:
            raise ValueError("Metadata channel count does not match the samples")

    @property
    def duration_seconds(self) -> float:
        """Return the audio duration in seconds."""

        return self.metadata.duration_seconds


@dataclass(frozen=True)
class VoiceActivitySegment:
    """A contiguous interval classified as active speech."""

    start_time: float
    end_time: float
    confidence: float

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("Voice activity intervals must have positive duration")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Voice activity confidence must be in the range [0, 1]")


@dataclass(frozen=True)
class AudioInput:
    """Canonical input description for either supported ingestion mode."""

    session_id: str
    source_path: Path | None
    audio_path: Path | None
    transcript_path: Path | None
    source_type: SourceType
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MeetTranscriptEntry:
    """A normalized, timestamped speech entry from a Google Meet transcript."""

    entry_id: str
    speaker_id: str
    speaker_name: str
    start_time: float
    end_time: float
    text: str
    language_code: str | None = None

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("Transcript entries must have positive timestamps")
        if not self.text.strip():
            raise ValueError("Transcript entries must contain text")


@dataclass(frozen=True)
class SpeakerSegment:
    """A timeline segment ready to become an AI-layer audio clip."""

    segment_id: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    speaker_source: SpeakerSource
    is_overlap: bool = False
    clip_path: Path | None = None

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("Speaker segments must have positive timestamps")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Speaker confidence must be in the range [0, 1]")


@dataclass(frozen=True)
class TranscriptParseResult:
    """Parsed Meet transcript entries plus non-fatal data-quality warnings."""

    entries: list[MeetTranscriptEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""
