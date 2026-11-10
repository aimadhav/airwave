"""Provider contracts used by the RAG layer."""

from collections.abc import Sequence
from typing import Protocol

from .models import AnswerExample, RetrievalQuery, RetrievalResult


class AnswerCorpus(Protocol):
    """Source of normalized historical answer examples."""

    def load(self) -> Sequence[AnswerExample]:
        """Return normalized examples."""


class Retriever(Protocol):
    """Search contract independent of the indexing implementation."""

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Return ranked, diverse examples."""
