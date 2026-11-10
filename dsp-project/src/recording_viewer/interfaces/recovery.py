"""Narrow, mockable boundaries for optional local model providers."""

from typing import Protocol
import numpy as np
from recording_viewer.models import TranscriptWord
from recording_viewer.models.recovery import SeparationOutput


class SpeechSeparator(Protocol):
    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparationOutput:
        """Return time-synchronous individual voices, not enhanced mixed audio."""
        ...


class VoiceEmbedder(Protocol):
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a finite, nonzero speaker vector or a typed provider error."""
        ...


class ArrayTranscriber(Protocol):
    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptWord]:
        """Return clip-relative word times with optional recognition probabilities."""
        ...
