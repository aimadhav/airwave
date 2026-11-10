"""Deterministic backend for model-provider tests with no external assets."""

import numpy as np
from dsp.diarization.interfaces.inference import ModelSpeakerInterval


class MockSpeakerBackend:
    def __init__(self, intervals: list[ModelSpeakerInterval]) -> None:
        self.intervals = intervals

    def infer(self, samples: np.ndarray, sample_rate: int) -> list[ModelSpeakerInterval]:
        return list(self.intervals)
