"""Explicit deterministic providers for offline integration tests, not runtime."""

import numpy as np
from recording_viewer.models import TranscriptWord
from recording_viewer.models.recovery import SeparationOutput


class MockSpeechSeparator:
    def __init__(self, tracks: tuple[np.ndarray, ...]) -> None:
        self.tracks = tracks

    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparationOutput:
        return SeparationOutput(tuple(t.copy() for t in self.tracks), sample_rate)


class MockVoiceEmbedder:
    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.vectors = iter(vectors)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return next(self.vectors).copy()


class MockArrayTranscriber:
    def __init__(self, transcripts: list[list[TranscriptWord]]) -> None:
        self.transcripts = iter(transcripts)

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptWord]:
        return list(next(self.transcripts))
