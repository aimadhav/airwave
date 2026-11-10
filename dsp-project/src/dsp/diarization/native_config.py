"""Explicit local configuration for the optional NeMo-Speech.cpp adapter."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from dsp.exceptions import DiarizationConfigurationError


@dataclass(frozen=True)
class NativeDiarizationConfig:
    """No automatic downloads; both runtime and weights must already exist."""

    executable: str
    model: str
    preset: str
    full_attention: bool
    max_full_attention_seconds: float
    timeout_seconds: float
    onset: float | None = None
    offset: float | None = None
    pad_onset_seconds: float | None = None
    pad_offset_seconds: float | None = None
    min_duration_on: float | None = None
    min_duration_off: float | None = None

    def __post_init__(self) -> None:
        if self.preset not in {"v3-offline", "v3-streaming"}:
            raise DiarizationConfigurationError("Native V3 preset must be v3-offline or v3-streaming")
        if not isinstance(self.full_attention, bool):
            raise DiarizationConfigurationError("full_attention must be a boolean")
        for name in ("timeout_seconds", "max_full_attention_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise DiarizationConfigurationError(f"{name} must be finite and positive")
        for name in ("onset", "offset", "pad_onset_seconds", "pad_offset_seconds", "min_duration_on", "min_duration_off"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise DiarizationConfigurationError(f"{name} must be finite and non-negative")
                if name in {"onset", "offset"} and not 0 < value < 1:
                    raise DiarizationConfigurationError(f"{name} must be between zero and one")
        if self.onset is not None and self.offset is not None and self.offset > self.onset:
            raise DiarizationConfigurationError("offset cannot exceed onset")
        if any(not isinstance(value, str) or not value for value in (self.executable, self.model)):
            raise DiarizationConfigurationError("Explicit native runtime and model paths are required")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "NativeDiarizationConfig":
        """Read versioned runtime defaults externally and apply explicit overrides."""
        defaults = Path(__file__).resolve().parents[3] / "configs/native_diarization_config.json"
        try:
            configured = json.loads(defaults.read_text(encoding="utf-8"))
            return cls(**{**configured, **values})
        except (OSError, TypeError, ValueError) as error:
            raise DiarizationConfigurationError(f"Invalid native diarization configuration: {error}") from error

    @classmethod
    def load_defaults(cls) -> "NativeDiarizationConfig":
        """Create typed defaults without embedding model identifiers in source."""
        return cls.from_mapping({})

    def validate_assets(self) -> None:
        """Reject missing files instead of invoking the runtime's model downloader."""
        for filename in (self.executable, self.model):
            if not Path(filename).is_file():
                raise DiarizationConfigurationError(
                    f"Missing native diarization asset: {filename}. See Documentation/DSP/09_Clear_Meeting_Diarization_Comparison.md."
                )
