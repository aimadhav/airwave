"""Canonical lab request/response DTOs; independent of any speech runtime."""

from dataclasses import dataclass, field
import math
import re
from typing import Literal
from overlap_lab.exceptions import LabInputError


@dataclass(frozen=True)
class ClipSpec:
    id: str
    start: float
    end: float
    speakers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", self.id):
            raise LabInputError("Invalid clip identifier")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (self.start, self.end)) or not 0 <= self.start < self.end:
            raise LabInputError("Clip times must be finite, nonnegative and increasing")
        if len(self.speakers) != 2 or len(set(self.speakers)) != 2 or any(not isinstance(s, str) for s in self.speakers):
            raise LabInputError("Select exactly two distinct speaker IDs")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ClipSpec":
        try:
            return cls(str(values["id"]), values["start"], values["end"], tuple(values["speakers"]))
        except (KeyError, TypeError) as error:
            raise LabInputError("Invalid clip specification") from error


@dataclass(frozen=True)
class ReferenceAnnotation:
    clip: ClipSpec
    texts: dict[str, str]
    reviewed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.reviewed, bool) or set(self.texts) != set(self.clip.speakers) or any(not isinstance(t, str) for t in self.texts.values()):
            raise LabInputError("References must have one text per selected speaker and an explicit review state")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ReferenceAnnotation":
        try:
            return cls(ClipSpec.from_mapping(values["clip"]), dict(values["texts"]), values["reviewed"])
        except (KeyError, TypeError, ValueError) as error:
            raise LabInputError("Invalid reference annotation") from error


@dataclass(frozen=True)
class ExperimentRequest:
    recording_id: str
    reference: ReferenceAnnotation
    variants: tuple[str, ...]
    context_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.recording_id, str) or not re.fullmatch(r"current|[a-f0-9]{32}", self.recording_id):
            raise LabInputError("Unknown recording identifier")
        if not self.variants or len(set(self.variants)) != len(self.variants) or any(not isinstance(v, str) for v in self.variants):
            raise LabInputError("Choose distinct experiment variants")
        if isinstance(self.context_seconds, bool) or not isinstance(self.context_seconds, (int, float)) or not math.isfinite(self.context_seconds) or self.context_seconds < 0:
            raise LabInputError("Context must be finite and nonnegative")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ExperimentRequest":
        try:
            return cls(values["recording_id"], ReferenceAnnotation.from_mapping(values["reference"]),
                       tuple(values["variants"]), values["context_seconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise LabInputError("Invalid experiment request") from error


@dataclass(frozen=True)
class LabJob:
    id: str
    recording_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str
    result: dict[str, object] | None = None


@dataclass(frozen=True)
class WordErrorScore:
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float | None
    alignment: tuple[dict[str, str], ...] = field(default_factory=tuple)
