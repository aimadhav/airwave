"""External, validated policy for conservative overlap recovery."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from recording_viewer.exceptions import RecoveryError


@dataclass(frozen=True)
class RecoveryConfig:
    """Thresholds are engineering gates, not calibrated correctness probabilities."""

    enabled: bool
    backend: str
    context_seconds: float
    max_core_seconds: float
    minimum_overlap_seconds: float
    max_windows: int
    reference_seconds: float
    reference_guard_seconds: float
    max_references: int
    minimum_references: int
    minimum_embedding_seconds: float
    minimum_cosine: float
    minimum_cosine_margin: float
    maximum_stream_correlation: float
    minimum_stream_rms: float
    minimum_energy_ratio: float
    minimum_reconstruction_db: float
    minimum_word_probability: float
    minimum_mean_probability: float
    maximum_duplicate_similarity: float
    minimum_duplicate_words: int
    embedding_model: str
    embedding_threads: int
    separator_python: str
    separator_directory: str
    separator_revision: str
    separator_repository: str
    separator_sample_rate: int
    separator_threads: int
    separator_timeout_seconds: float
    separator_gain_mode: str = "raw"
    separator_peak_limit: float = 0.98

    def __post_init__(self) -> None:
        if self.separator_gain_mode not in {"raw", "shared_peak"}:
            raise RecoveryError("Invalid separator gain mode")
        if (isinstance(self.separator_peak_limit, bool) or not isinstance(self.separator_peak_limit, (int, float))
                or not math.isfinite(self.separator_peak_limit) or not 0 < self.separator_peak_limit <= 1):
            raise RecoveryError("Separator peak limit must lie in (0, 1]")
        for name in ("reference_guard_seconds", "minimum_cosine", "minimum_reconstruction_db"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RecoveryError(f"{name} must be a finite number")
        if not isinstance(self.enabled, bool) or self.backend not in {"nemo_native", "sherpa"}:
            raise RecoveryError("Invalid recovery enable switch or timeline backend")
        for name in ("context_seconds", "max_core_seconds", "minimum_overlap_seconds", "reference_seconds",
                     "minimum_embedding_seconds", "minimum_stream_rms", "separator_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RecoveryError(f"{name} must be finite and positive")
        if not math.isfinite(self.reference_guard_seconds) or self.reference_guard_seconds < 0:
            raise RecoveryError("Reference guard must be finite and nonnegative")
        for name in ("minimum_cosine_margin", "maximum_stream_correlation", "minimum_energy_ratio",
                     "minimum_word_probability", "minimum_mean_probability", "maximum_duplicate_similarity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise RecoveryError(f"{name} must lie in (0, 1]")
        if not math.isfinite(self.minimum_cosine) or not -1 <= self.minimum_cosine <= 1:
            raise RecoveryError("Invalid cosine threshold")
        if not math.isfinite(self.minimum_reconstruction_db):
            raise RecoveryError("Invalid reconstruction threshold")
        for name in ("max_windows", "max_references", "minimum_references", "minimum_duplicate_words",
                     "embedding_threads", "separator_threads", "separator_sample_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RecoveryError(f"{name} must be a positive integer")
        if self.minimum_references > self.max_references or self.minimum_overlap_seconds > self.max_core_seconds:
            raise RecoveryError("Inconsistent recovery/reference limits")
        for name in ("embedding_model", "separator_python", "separator_directory", "separator_revision", "separator_repository"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise RecoveryError(f"{name} must be explicit")

    @classmethod
    def load(cls, path: Path) -> "RecoveryConfig":
        """Reject malformed policy before any inference begins."""
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError) as error:
            raise RecoveryError(f"Invalid overlap recovery configuration: {error}") from error
