"""Local adapter for Upsckata-style ``bank.json`` and ``answers.json``."""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from ..exceptions import DatasetError
from ..models import AnswerExample


_DIRECTIVES = (
    "critically examine", "critically analyse", "critically analyze",
    "evaluate", "analyse", "analyze", "examine", "discuss", "comment",
    "justify", "elucidate", "assess", "enumerate",
)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _directive(question: str) -> str:
    lowered = question.lower()
    for candidate in _DIRECTIVES:
        if candidate in lowered:
            return candidate
    return "unspecified"


def _freshness(year: int | None, current_year: int) -> str:
    if year is None:
        return "possibly_stale"
    age = current_year - year
    if age <= 2:
        return "time_sensitive"
    if age <= 6:
        return "historical"
    return "possibly_stale"


def load_upsckata_corpus(
    bank_path: str | Path,
    answers_path: str | Path,
    *,
    current_year: int,
) -> list[AnswerExample]:
    """Join and normalize Upsckata JSON files by question ID.

    The loader rejects orphan records because silently mismatching question and
    answer IDs would produce unsafe RAG citations.
    """

    try:
        bank = json.loads(Path(bank_path).read_text(encoding="utf-8"))
        answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Unable to read corpus files: {exc}") from exc

    questions = bank.get("questions") if isinstance(bank, dict) else None
    if not isinstance(questions, list) or not isinstance(answers, dict):
        raise DatasetError("Expected bank.questions array and answers object")

    question_by_id = {str(row.get("id")): row for row in questions if row.get("id") is not None}
    answer_ids = set(answers)
    question_ids = set(question_by_id)
    if answer_ids != question_ids:
        missing_answers = sorted(question_ids - answer_ids)[:5]
        missing_questions = sorted(answer_ids - question_ids)[:5]
        raise DatasetError(
            f"Corpus IDs do not match; missing_answers={missing_answers}, "
            f"missing_questions={missing_questions}"
        )

    result: list[AnswerExample] = []
    for identifier, question_row in question_by_id.items():
        answer_row = answers[identifier]
        question = str(question_row.get("text", "")).strip()
        answer_text = str(answer_row.get("answer", "")).strip()
        if not question or not answer_text:
            continue
        year = _as_int(question_row.get("year"))
        source_page = _as_int(question_row.get("page"))
        rank = _as_int(question_row.get("rank"))
        marks = _as_int(question_row.get("marks"))
        word_limit = _as_int(question_row.get("words"))
        result.append(
            AnswerExample(
                id=f"upsckata-{identifier}",
                question=question,
                answer_text=answer_text,
                approach_summary=str(answer_row.get("approach", "")).strip(),
                paper=str(question_row.get("paper") or "Unknown"),
                topic=str(question_row.get("subject") or "General Studies"),
                directive=_directive(question),
                marks=marks,
                word_limit=word_limit,
                topper=str(question_row.get("topper") or "Unknown"),
                rank=rank,
                exam_year=year,
                source_url=str(question_row.get("url") or ""),
                source_page=source_page,
                ocr_status="transcribed",
                ocr_confidence=None,
                last_verified=None,
                content_freshness=_freshness(year, current_year),
                tags=tuple(sorted(set(re.findall(r"[a-z]{3,}", question.lower()))))[:20],
            )
        )
    return result
