"""Diarization provider abstractions and deterministic fallbacks."""

from dsp.diarization.providers import (
    DiarizationProvider,
    MockDiarizer,
    SingleSpeakerDiarizer,
)

__all__ = ["DiarizationProvider", "MockDiarizer", "SingleSpeakerDiarizer"]
