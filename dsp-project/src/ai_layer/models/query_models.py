from dataclasses import dataclass, field
from typing import Literal

from ai_layer.models.base_models import TimeRange

QueryIntentCategory = Literal["decision", "task", "fact", "general"]


@dataclass
class UserQuery:
    query_text: str
    target_session_id: str | None = None


@dataclass
class QueryIntent:
    intent_category: QueryIntentCategory = "general"
    target_person: str | None = None
    target_topic: str | None = None


@dataclass
class RetrievedSnippet:
    snippet_id: str
    content: str
    speaker_name: str
    time_range: TimeRange
    relevance_score: float = 0.0
    session_id: str = ""
    session_title: str = ""


@dataclass
class AnswerCitation:
    speaker_name: str
    timestamp_str: str
    verbatim_quote: str
    session_id: str = ""
    session_title: str = ""


@dataclass
class QueryAnswer:
    answer_text: str
    citations: list[AnswerCitation] = field(default_factory=list)
    is_found: bool = True


# -----------------------------------------------------------------------------
# Canonical Contract 3: Query & RAG Retrieval Interface
# -----------------------------------------------------------------------------


@dataclass
class SearchRequest:
    """Contract 3 Search Request payload."""

    workspace_id: str
    project_id: str | None = None
    query_text: str = ""
    query_embedding: list[float] = field(default_factory=list)
    top_k: int = 5
    include_superseded: bool = False
    walk_edges: bool = True


@dataclass
class ConnectedEdge:
    relation_type: str
    connected_content_id: str
    content: str


@dataclass
class SourceUtteranceInfo:
    utterance_id: str
    speaker_name: str
    start_ts: float
    end_ts: float
    raw_text: str


@dataclass
class ContractRetrievedResult:
    content_id: str
    type: str
    content: str
    similarity_score: float = 0.0
    recency_score: float = 1.0
    combined_rank_score: float = 0.0
    source_utterances: list[SourceUtteranceInfo] = field(default_factory=list)
    connected_edges: list[ConnectedEdge] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Contract 3 Search Response with provenance and rank scores."""

    results: list[ContractRetrievedResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "results": [
                {
                    "content_id": r.content_id,
                    "type": r.type,
                    "content": r.content,
                    "similarity_score": r.similarity_score,
                    "recency_score": r.recency_score,
                    "combined_rank_score": r.combined_rank_score,
                    "source_utterances": [
                        {
                            "utterance_id": u.utterance_id,
                            "speaker_name": u.speaker_name,
                            "start_ts": u.start_ts,
                            "end_ts": u.end_ts,
                            "raw_text": u.raw_text,
                        }
                        for u in r.source_utterances
                    ],
                    "connected_edges": [
                        {
                            "relation_type": e.relation_type,
                            "connected_content_id": e.connected_content_id,
                            "content": e.content,
                        }
                        for e in r.connected_edges
                    ],
                }
                for r in self.results
            ]
        }
