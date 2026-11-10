"""Deterministic timestamped transcription for isolated verification."""

from pathlib import Path
from recording_viewer.models import TranscriptWord


class MockTimestampedTranscriber:
    """Fixture-only provider, never presented as real recognition."""

    def __init__(self, words: list[TranscriptWord]) -> None:
        self._words = list(words)

    def transcribe(self, audio_path: Path) -> list[TranscriptWord]:
        """Return supplied fixture words without external dependencies."""
        return list(self._words)
