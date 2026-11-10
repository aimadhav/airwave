"""Typed, external runtime configuration for private recording exports."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from recording_viewer.exceptions import RecordingViewerError


@dataclass(frozen=True)
class RecordingViewerConfig:
    """Settings are explicit; recognition never auto-downloads weights."""

    model_repository: str
    model_directory: str
    model_revision: str
    device: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    vad_filter: bool
    condition_on_previous_text: bool
    language: str
    minimum_word_coverage: float
    waveform_bins: int
    decode_timeout_seconds: float
    session_id: str
    output_directory: str
    source: str
    max_duration_seconds: float | None = None
    recovery_config: str | None = None
    default_backend: str = "nemo_native"
    maximum_no_speech_probability: float = 0.6
    minimum_segment_log_probability: float = -1.0
    maximum_compression_ratio: float = 2.4
    dsp_config_path: str | None = None

    def __post_init__(self) -> None:
        if self.default_backend not in {"sherpa", "nemo_native"}:
            raise RecordingViewerError("Invalid default timeline backend")
        if (not math.isfinite(self.maximum_no_speech_probability) or not 0 <= self.maximum_no_speech_probability <= 1
                or not math.isfinite(self.minimum_segment_log_probability) or self.minimum_segment_log_probability > 0
                or not math.isfinite(self.maximum_compression_ratio) or self.maximum_compression_ratio <= 0):
            raise RecordingViewerError("Invalid recognition quality thresholds")
        if self.recovery_config is not None and (not isinstance(self.recovery_config, str) or not self.recovery_config):
            raise RecordingViewerError("Recovery config must be an explicit path or null")
        if self.dsp_config_path is not None and (not isinstance(self.dsp_config_path, str) or not self.dsp_config_path):
            raise RecordingViewerError("DSP config must be an explicit path or null")
        if not 0 < self.minimum_word_coverage <= 1:
            raise RecordingViewerError("Invalid minimum word coverage")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (self.cpu_threads, self.beam_size, self.waveform_bins)):
            raise RecordingViewerError("Thread, beam and waveform counts must be positive integers")
        if not math.isfinite(self.decode_timeout_seconds) or self.decode_timeout_seconds <= 0:
            raise RecordingViewerError("Decoder timeout must be positive")
        if not isinstance(self.vad_filter, bool) or not isinstance(self.condition_on_previous_text, bool):
            raise RecordingViewerError("Recognition switches must be booleans")
        if self.max_duration_seconds is not None and (not math.isfinite(self.max_duration_seconds) or self.max_duration_seconds <= 0):
            raise RecordingViewerError("Maximum recording duration must be finite and positive")

    @classmethod
    def load(cls, path: Path) -> "RecordingViewerConfig":
        """Apply an optional ignored local override, then validate the full policy."""
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            local_path = path.with_suffix(".local.json")
            if local_path.is_file():
                values = {**values, **json.loads(local_path.read_text(encoding="utf-8"))}
            return cls(**values)
        except (OSError, TypeError, ValueError) as error:
            raise RecordingViewerError(f"Invalid recording-viewer configuration: {error}") from error
