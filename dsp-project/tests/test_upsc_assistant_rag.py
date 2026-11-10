"""Tests for the UPSC RAG ingestion and retrieval vertical slice."""

import json

import pytest

from upsc_assistant.adapters.json_corpus import load_upsckata_corpus
from upsc_assistant.exceptions import DatasetError
from upsc_assistant.models import AnswerExample, RetrievalQuery
from upsc_assistant.services.retrieval import InMemoryRetriever


def _write_corpus(tmp_path, *, mismatch=False):
    questions = [{
        "id": 1, "text": "Discuss federalism and cooperative governance", "marks": 10,
        "words": 150, "paper": "GS2", "subject": "Polity", "topper": "A", "rank": 1,
        "year": 2024, "page": 3, "url": "https://example.test/a.pdf#page=3",
    }, {
        "id": 2, "text": "Evaluate the role of ethics in public service", "marks": 10,
        "words": 150, "paper": "GS4", "subject": "Ethics", "topper": "B", "rank": 2,
        "year": 2018, "page": 4, "url": "https://example.test/b.pdf#page=4",
    }]
    answers = {
        "1": {"answer": "Federalism requires cooperation between Union and states.", "approach": "Defines the idea and gives examples."},
        "2": {"answer": "Ethics strengthens accountability in public service.", "approach": "Defines ethics and applies it to administration."},
    }
    if mismatch:
        answers.pop("2")
    bank_path = tmp_path / "bank.json"
    answers_path = tmp_path / "answers.json"
    bank_path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    answers_path.write_text(json.dumps(answers), encoding="utf-8")
    return bank_path, answers_path


def test_loader_joins_records_and_marks_freshness(tmp_path):
    bank, answers = _write_corpus(tmp_path)
    corpus = load_upsckata_corpus(bank, answers, current_year=2026)
    assert len(corpus) == 2
    assert corpus[0].id == "upsckata-1"
    assert corpus[0].directive == "discuss"
    assert corpus[0].content_freshness == "time_sensitive"
    assert corpus[1].content_freshness == "possibly_stale"


def test_loader_rejects_orphan_ids(tmp_path):
    bank, answers = _write_corpus(tmp_path, mismatch=True)
    with pytest.raises(DatasetError, match="IDs do not match"):
        load_upsckata_corpus(bank, answers, current_year=2026)


def test_retriever_applies_filters_and_returns_explainable_matches(tmp_path):
    bank, answers = _write_corpus(tmp_path)
    corpus = load_upsckata_corpus(bank, answers, current_year=2026)
    results = InMemoryRetriever(corpus).search(
        RetrievalQuery(text="cooperative federalism", paper="GS2", limit=3)
    )
    assert len(results) == 1
    assert results[0].example.id == "upsckata-1"
    assert "federalism" in results[0].matched_terms
    assert results[0].score > 0


def test_retriever_requires_query_text():
    example = AnswerExample(
        id="x", question="q", answer_text="a", approach_summary="", paper="GS2",
        topic="Polity", directive="discuss", marks=10, word_limit=150, topper="A",
        rank=1, exam_year=2024, source_url="", source_page=1, ocr_status="transcribed",
        ocr_confidence=None, last_verified=None, content_freshness="historical",
    )
    with pytest.raises(Exception, match="cannot be empty"):
        InMemoryRetriever([example]).search(RetrievalQuery(text=" "))
