"""External policy for a deliberately small, offline duration-filter ablation."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from overlap_lab.exceptions import LabInputError


@dataclass(frozen=True)
class DetectionEvaluationConfig:
    minimum_durations: tuple[float, ...]
    held_out_fraction: float
    split_seed: str
    minimum_calibration_positives: int
    output_directory: str
    contexts: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.minimum_durations or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or v <= 0 for v in self.minimum_durations):
            raise LabInputError("Detection thresholds must be finite and positive")
        if sorted(set(self.minimum_durations)) != list(self.minimum_durations):
            raise LabInputError("Detection thresholds must be sorted and unique")
        if (isinstance(self.held_out_fraction, bool) or not isinstance(self.held_out_fraction, (int, float))
                or not math.isfinite(self.held_out_fraction) or not 0 < self.held_out_fraction < 1):
            raise LabInputError("Holdout fraction must lie between zero and one")
        if type(self.minimum_calibration_positives) is not int or self.minimum_calibration_positives < 1:
            raise LabInputError("At least one calibration positive is required")
        if not isinstance(self.split_seed, str) or not self.split_seed.strip() or not isinstance(self.output_directory, str) or not self.output_directory.strip() or not self.contexts or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in self.contexts):
            raise LabInputError("Explicit split, output directory and valid contexts are required")

    @classmethod
    def load(cls, path: Path) -> "DetectionEvaluationConfig":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["minimum_durations"] = tuple(data["minimum_durations"])
            data["contexts"] = tuple(data["contexts"])
            return cls(**data)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise LabInputError("Invalid detection evaluation policy") from error
