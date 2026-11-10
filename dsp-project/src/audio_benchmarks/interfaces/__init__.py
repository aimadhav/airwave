"""Narrow provider boundaries for isolated benchmark execution."""

from pathlib import Path
from typing import Protocol
from audio_benchmarks.models import DiarizationPrediction


class DiarizationProvider(Protocol):
    def predict(self, audio_path: Path, output_directory: Path, experiment_id: str) -> DiarizationPrediction:
        """Return canonical intervals and measured wall time, or a typed error."""
        ...
