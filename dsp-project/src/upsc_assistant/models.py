"""Canonical contracts for the UPSC retrieval layer."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Paper = Literal["GS1", "GS2", "GS3", "GS4", "Essay"]
ContentFreshness = Literal["evergreen", "historical", "time_sensitive", "possibly_stale"]


@dataclass(frozen=True)
class AnswerExample:
    """A normalized historical answer with provenance and retrieval metadata."""

    id: str
    question: str
    answer_text: str
    approach_summary: str
    paper: str
    topic: str
    directive: str
    marks: int | None
    word_limit: int | None
    topper: str
    rank: int | None
    exam_year: int | None
    source_url: str
    source_page: int | None
    ocr_status: str
    ocr_confidence: float | None
    last_verified: date | None
    content_freshness: ContentFreshness
    content_type: str = "historical_answer_example"
    source: str = "upsckata"
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RetrievalQuery:
    """A retrieval request with optional hard filters."""

    text: str
    paper: str | None = None
    directive: str | None = None
    marks: int | None = None
    word_limit: int | None = None
    topic: str | None = None
    limit: int = 5


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked result with explainable matching signals."""

    example: AnswerExample
    score: float
    keyword_score: float
    semantic_score: float
    matched_terms: tuple[str, ...]
