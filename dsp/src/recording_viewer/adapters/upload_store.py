"""Filesystem recording uploads, restricted to an explicit private root."""

from pathlib import Path
import re
from typing import BinaryIO
from recording_viewer.exceptions import UploadInputError
from recording_viewer.models.upload import StoredRecording


class FileRecordingUploadStore:
    """Use server-generated IDs for directories and sanitized original names."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, job_id: str, filename: str, stream: BinaryIO, length: int) -> StoredRecording:
        """Atomically accept complete content; a partial file is never processed."""
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise UploadInputError("Invalid upload identifier")
        directory = self.root / job_id
        directory.mkdir(parents=True, exist_ok=False)
        source = directory / filename
        if source.resolve().parent != directory.resolve():
            raise UploadInputError("Invalid recording filename")
        partial = directory / "upload.partial"
        remaining = length
        with partial.open("xb") as output:
            while remaining:
                block = stream.read(min(64 * 1024, remaining))
                if not block:
                    raise UploadInputError("Upload interrupted before the complete file arrived")
                output.write(block)
                remaining -= len(block)
        partial.rename(source)
        return StoredRecording(source=source, result=directory / "result")
