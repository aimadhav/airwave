"""Validated configuration for optional CPU multi-speaker inference."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

from dsp.exceptions import DiarizationConfigurationError
from dsp.diarization.native_config import NativeDiarizationConfig


@dataclass(frozen=True)
class LocalDiarizationConfig:
    backend: str = "sherpa"
    native: NativeDiarizationConfig = field(default_factory=NativeDiarizationConfig.load_defaults)
    segmentation_model: str = "models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_model: str = "models/diarization/nemo_en_titanet_small.onnx"
    num_speakers: int = -1
    cluster_threshold: float = 0.9
    num_threads: int = 2
    window_shift_ratio: float = 0.1
    min_duration_on: float = 0.2
    min_duration_off: float = 0.3

    def __post_init__(self) -> None:
        if self.backend not in {"sherpa", "nemo_native"}:
            raise DiarizationConfigurationError("Unknown local diarization backend")
        if not isinstance(self.native, NativeDiarizationConfig):
            raise DiarizationConfigurationError("native must be a NativeDiarizationConfig")
        if self.backend == "nemo_native" and self.num_speakers != -1:
            raise DiarizationConfigurationError("Nemotron predicts speaker activity automatically; forced counts are unsupported")
        if not isinstance(self.num_speakers, int) or isinstance(self.num_speakers, bool):
            raise DiarizationConfigurationError("num_speakers must be an integer")
        if not isinstance(self.num_threads, int) or isinstance(self.num_threads, bool):
            raise DiarizationConfigurationError("num_threads must be an integer")
        if self.num_speakers != -1 and self.num_speakers < 1:
            raise DiarizationConfigurationError("num_speakers must be -1 (automatic) or a positive count")
        if not math.isfinite(self.cluster_threshold) or not 0 < self.cluster_threshold < 2:
            raise DiarizationConfigurationError("cluster_threshold must lie between 0 and 2")
        if self.num_threads < 1 or not 0 < self.window_shift_ratio <= 1:
            raise DiarizationConfigurationError("Invalid thread count or segmentation window shift")
        if any(not math.isfinite(v) or v < 0 for v in (self.min_duration_on, self.min_duration_off)):
            raise DiarizationConfigurationError("Segment durations must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "LocalDiarizationConfig":
        """Load the local_model section without accepting unknown parameters."""
        try:
            normalized = dict(values)
            if "native" in normalized:
                normalized["native"] = NativeDiarizationConfig.from_mapping(normalized["native"])
            return cls(**normalized)
        except (TypeError, ValueError) as error:
            raise DiarizationConfigurationError(f"Invalid local_model configuration: {error}") from error

    def validate_assets(self) -> None:
        """Model files are explicit local inputs; inference never downloads them."""
        if self.backend == "nemo_native":
            self.native.validate_assets()
            return
        for filename in (self.segmentation_model, self.embedding_model):
            if not Path(filename).is_file():
                raise DiarizationConfigurationError(
                    f"Missing diarization model: {filename}. Run python scripts/setup_diarization.py first."
                )
