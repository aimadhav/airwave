"""Source-grounded retrieval for the UPSC answer-writing assistant."""

from .models import AnswerExample, RetrievalQuery, RetrievalResult
from .services.retrieval import InMemoryRetriever

__all__ = ["AnswerExample", "RetrievalQuery", "RetrievalResult", "InMemoryRetriever"]
