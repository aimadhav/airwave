"""Deterministic hybrid retrieval suitable for local development and tests."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from ..exceptions import RetrievalError
from ..models import AnswerExample, RetrievalQuery, RetrievalResult


_STOP_WORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "what",
    "how", "does", "can", "should", "into", "their", "about", "between",
}


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOP_WORDS]


class InMemoryRetriever:
    """TF-IDF-like lexical search plus metadata filtering and diversity.

    This is intentionally provider-free. A vector adapter can implement the
    same ``Retriever`` contract later without changing the services above it.
    """

    def __init__(self, examples: Iterable[AnswerExample]) -> None:
        self._examples = list(examples)
        self._documents: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        for example in self._examples:
            terms = Counter(_tokens(" ".join((example.question, example.answer_text, example.approach_summary))))
            self._documents[example.id] = terms
            self._document_frequency.update(terms.keys())
        self._by_id = {example.id: example for example in self._examples}

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Return filtered, ranked, and topper-diverse results."""

        if not query.text.strip():
            raise RetrievalError("Retrieval text cannot be empty")
        if query.limit < 1:
            raise RetrievalError("Retrieval limit must be positive")

        query_terms = Counter(_tokens(query.text))
        candidate_ids = [
            example.id for example in self._examples
            if self._matches_filters(example, query)
        ]
        scored = [self._score(self._by_id[item], query_terms) for item in candidate_ids]
        scored.sort(key=lambda result: (-result.score, result.example.id))

        selected: list[RetrievalResult] = []
        topper_counts: defaultdict[str, int] = defaultdict(int)
        for result in scored:
            topper = result.example.topper.casefold()
            if topper_counts[topper] >= 2 and len(selected) < query.limit:
                continue
            selected.append(result)
            topper_counts[topper] += 1
            if len(selected) >= query.limit:
                break
        return selected

    @staticmethod
    def _matches_filters(example: AnswerExample, query: RetrievalQuery) -> bool:
        if query.paper and example.paper.casefold() != query.paper.casefold():
            return False
        if query.directive and example.directive.casefold() != query.directive.casefold():
            return False
        if query.topic and query.topic.casefold() not in example.topic.casefold():
            return False
        if query.marks is not None and example.marks != query.marks:
            return False
        if query.word_limit is not None and example.word_limit != query.word_limit:
            return False
        return True

    def _score(self, example: AnswerExample, query_terms: Counter[str]) -> RetrievalResult:
        document = self._documents[example.id]
        matched = tuple(sorted(set(query_terms) & set(document)))
        if not matched:
            return RetrievalResult(example, 0.0, 0.0, 0.0, matched)
        total_documents = max(len(self._examples), 1)
        semantic = 0.0
        for term in matched:
            inverse_frequency = math.log((1 + total_documents) / (1 + self._document_frequency[term])) + 1
            semantic += min(query_terms[term], document[term]) * inverse_frequency
        semantic /= max(sum(query_terms.values()), 1)
        keyword = len(matched) / max(len(set(query_terms)), 1)
        score = (0.65 * semantic) + (0.35 * keyword)
        return RetrievalResult(example, score, keyword, semantic, matched)
