"""Upload job DTOs shared by service, adapters and HTTP presentation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class StoredRecording:
    """An uploaded file and its isolated future result directory."""

    source: Path
    result: Path


@dataclass(frozen=True)
class UploadJob:
    """Public job state; never exposes raw logs or arbitrary filesystem paths."""

    id: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed"]
    message: str
    result_url: str | None = None
