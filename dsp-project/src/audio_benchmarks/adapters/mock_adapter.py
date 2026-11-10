"""Deterministic provider for isolated tests without installed speech models."""

from pathlib import Path
from audio_benchmarks.models import DiarizationPrediction, Interval


class MockDiarizationProvider:
    def __init__(self, intervals: tuple[Interval,...] = ()) -> None:
        self.intervals = intervals

    def predict(self, audio_path: Path, output_directory: Path, experiment_id: str) -> DiarizationPrediction:
        """Honor the production protocol without filesystem or inference dependencies."""
        return DiarizationPrediction(self.intervals,0.0)
