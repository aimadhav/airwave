"""Typed trusted runtime policy, never supplied over HTTP."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from overlap_lab.exceptions import LabInputError


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    directory: str
    repository: str


@dataclass(frozen=True)
class SeparatorOption:
    id: str
    label: str
    provider: str


@dataclass(frozen=True)
class LabConfig:
    storage_directory: str
    python_executable: str
    viewer_config: str
    recovery_config: str
    request_limit_bytes: int
    request_timeout_seconds: float
    max_clip_seconds: float
    max_context_seconds: float
    default_context_seconds: float
    job_timeout_seconds: float
    poll_interval_ms: int
    max_pending_jobs: int
    max_history: int
    max_annotations: int
    max_reference_characters: int
    max_reference_words: int
    listening_sample_size: int
    listening_minimum_seconds: float
    listening_voice_seconds: float
    models: tuple[ModelOption, ...]
    separators: tuple[SeparatorOption, ...]

    def __post_init__(self) -> None:
        for name in ("request_limit_bytes", "request_timeout_seconds", "max_clip_seconds", "max_context_seconds", "job_timeout_seconds",
                     "poll_interval_ms", "max_pending_jobs", "max_history", "max_annotations", "max_reference_characters", "max_reference_words",
                     "listening_sample_size", "listening_minimum_seconds", "listening_voice_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise LabInputError(f"Invalid lab policy: {name}")
        for name in ("request_limit_bytes", "poll_interval_ms", "max_pending_jobs", "max_history", "max_annotations", "max_reference_characters", "max_reference_words", "listening_sample_size"):
            if not isinstance(getattr(self, name), int):
                raise LabInputError(f"{name} must be an integer")
        if not 0 <= self.default_context_seconds <= self.max_context_seconds:
            raise LabInputError("Invalid default context")
        if self.listening_sample_size > self.max_annotations or self.listening_minimum_seconds > self.max_clip_seconds:
            raise LabInputError("Listening sample exceeds the configured limits")
        if not self.models or not self.separators or len({m.id for m in self.models}) != len(self.models):
            raise LabInputError("Model identifiers must be unique")

    @classmethod
    def load(cls, path: Path) -> "LabConfig":
        """Load only operator-controlled local configuration."""
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            values["models"] = tuple(ModelOption(**row) for row in values["models"])
            values["separators"] = tuple(SeparatorOption(**row) for row in values["separators"])
            return cls(**values)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise LabInputError(f"Cannot load lab configuration: {error}") from error
