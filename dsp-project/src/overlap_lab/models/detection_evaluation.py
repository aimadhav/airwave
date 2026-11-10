"""Pure binary detection counts, not transcription or speaker-ID accuracy."""

from dataclasses import dataclass
import math
from overlap_lab.exceptions import LabInputError


@dataclass(frozen=True)
class DetectionCase:
    """A reviewed detector candidate; absence of candidates is not evaluated."""

    id: str
    duration_seconds: float
    overlapping: bool

    def __post_init__(self) -> None:
        if (not isinstance(self.id, str) or not self.id or type(self.overlapping) is not bool
                or isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (int, float))
                or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0):
            raise LabInputError("Invalid labelled detection case")


@dataclass(frozen=True)
class DetectionCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
