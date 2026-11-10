"""Upload boundaries independent of model and storage implementations."""

from typing import BinaryIO, Protocol
from recording_viewer.models.upload import StoredRecording


class RecordingUploadStore(Protocol):
    """Store a bounded recording without using its filename as a path."""

    def save(self, job_id: str, filename: str, stream: BinaryIO, length: int) -> StoredRecording:
        """Persist exactly length bytes into an isolated directory."""
        ...


class RecordingProcessor(Protocol):
    """Build a recording page using explicit, local model configuration."""

    def process(self, job_id: str, recording: StoredRecording) -> None:
        """Return only after a completed result page exists, or raise an error."""
        ...
