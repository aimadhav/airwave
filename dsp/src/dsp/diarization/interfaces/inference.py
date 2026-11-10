"""Backend-neutral model output and inference contract."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class ModelSpeakerInterval:
    start: float
    end: float
    cluster: int


class SpeakerInferenceBackend(Protocol):
    def infer(self, samples: np.ndarray, sample_rate: int) -> list[ModelSpeakerInterval]:
        """Return model intervals in the original recording's time coordinate."""
        ...
