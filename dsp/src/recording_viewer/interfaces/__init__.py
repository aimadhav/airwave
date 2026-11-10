"""Narrow timestamped-recognition interface independent of runtime adapters."""

from pathlib import Path
from typing import Protocol
from recording_viewer.models import TranscriptWord


class TimestampedTranscriber(Protocol):
    """An adapter supplies recognized words independently of diarization."""

    def transcribe(self, audio_path: Path) -> list[TranscriptWord]:
        """Return words in the source audio's time coordinate system."""
        ...
