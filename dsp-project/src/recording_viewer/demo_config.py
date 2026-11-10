"""Explicit selection of the one podcast and experiment artifacts to share."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from recording_viewer.exceptions import RecordingViewerError


@dataclass(frozen=True)
class DemoExample:
    clip_id: str
    job_id: str

    def __post_init__(self) -> None:
        if (not isinstance(self.clip_id, str) or not isinstance(self.job_id, str)
                or not re.fullmatch(r"overlap_[0-9]{4}", self.clip_id) or not re.fullmatch(r"[a-f0-9]{32}", self.job_id)):
            raise RecordingViewerError("Invalid public-demo example selection")


@dataclass(frozen=True)
class PublicDemoConfig:
    recording_directory: str
    feedback_directory: str
    output_directory: str
    examples: tuple[DemoExample, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
                self.recording_directory, self.feedback_directory, self.output_directory)):
            raise RecordingViewerError("Explicit demo input and output directories are required")
        if not self.examples or len({example.clip_id for example in self.examples}) != len(self.examples):
            raise RecordingViewerError("Select distinct saved demo clips")

    @classmethod
    def load(cls, path: Path) -> "PublicDemoConfig":
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            values["examples"] = tuple(DemoExample(**row) for row in values["examples"])
            return cls(**values)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RecordingViewerError("Invalid public-demo export configuration") from error
