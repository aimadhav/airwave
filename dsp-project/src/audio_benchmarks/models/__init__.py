"""Canonical validated DTOs; independent of DSP/STT implementation classes."""

from dataclasses import dataclass
import math

from audio_benchmarks.exceptions import BenchmarkInputError


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                for v in (self.start, self.end)) or self.start < 0 or self.end <= self.start
                or not isinstance(self.speaker, str) or not self.speaker):
            raise BenchmarkInputError("Intervals require finite positive duration and a speaker label")


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    text: str

    def __post_init__(self) -> None:
        Interval(self.start, self.end, self.speaker)
        if not isinstance(self.text, str):
            raise BenchmarkInputError("Utterance text must be a string")


@dataclass(frozen=True)
class DiarizationPrediction:
    intervals: tuple[Interval, ...]
    elapsed_seconds: float
    clip_validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise BenchmarkInputError("Prediction runtime must be finite and nonnegative")
