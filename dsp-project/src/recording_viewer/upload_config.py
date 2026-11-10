"""Validated policy for the loopback-only recording upload service."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from recording_viewer.exceptions import RecordingViewerError


@dataclass(frozen=True)
class UploadConfig:
    """All limits and process paths are trusted configuration, never HTTP input."""

    max_upload_bytes: int
    max_pending_jobs: int
    request_timeout_seconds: float
    processing_timeout_seconds: float
    max_audio_seconds: float
    poll_interval_ms: int
    local_app_url: str
    python_executable: str
    base_config: str
    upload_directory: str
    allowed_extensions: list[str]

    def __post_init__(self) -> None:
        for value in (self.max_upload_bytes, self.max_pending_jobs, self.poll_interval_ms):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RecordingViewerError("Upload counts and byte limits must be positive integers")
        for value in (self.request_timeout_seconds, self.processing_timeout_seconds, self.max_audio_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise RecordingViewerError("Upload time limits must be finite and positive")
        location = urlsplit(self.local_app_url)
        if location.scheme != "http" or location.hostname not in {"127.0.0.1", "localhost"}:
            raise RecordingViewerError("The upload app must remain on loopback")
        if not self.allowed_extensions or any(not extension.startswith(".") or "/" in extension or "\\" in extension for extension in self.allowed_extensions):
            raise RecordingViewerError("Invalid upload extension policy")

    @classmethod
    def load(cls, path: Path) -> "UploadConfig":
        """Load policy without accepting unknown settings."""
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as error:
            raise RecordingViewerError(f"Invalid upload configuration: {error}") from error
