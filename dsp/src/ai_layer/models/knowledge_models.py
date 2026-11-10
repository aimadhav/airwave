from dataclasses import dataclass, field
from typing import Any, Literal

RelationshipType = Literal["novel", "reinforce", "supersede", "contradict", "elaborate"]


@dataclass
class KnowledgeRecord:
    record_id: str
    record_type: str  # 'fact', 'decision', 'action_item'
    statement: str
    session_id: str
    source_turn_id: str
    status: str = "active"  # 'active', 'superseded', 'flagged'
    related_record_ids: list[str] = field(default_factory=list)


@dataclass
class ChangeLogEntry:
    entry_id: str
    record_id: str
    action: str
    session_id: str
    details: str = ""


@dataclass
class KnowledgeRelationship:
    relationship_id: str
    source_record_id: str
    target_record_id: str
    relationship_type: str
    rationale: str = ""


@dataclass
class KnowledgeStore:
    records: list[KnowledgeRecord] = field(default_factory=list)
    change_log: list[ChangeLogEntry] = field(default_factory=list)
    relationships: list[KnowledgeRelationship] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Canonical Contract 2: AI Layer ➔ Database Layer (KnowledgePayload)
# -----------------------------------------------------------------------------


@dataclass
class UtterancePayload:
    utterance_id: str
    session_id: str
    speaker_id: str
    start_ts: float
    end_ts: float
    raw_text: str
    identity_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentUnitPayload:
    content_id: str
    workspace_id: str
    project_id: str | None
    type: str  # 'fact', 'decision', 'action_item'
    content: str
    embedding: list[float] = field(default_factory=list)
    status: str = "active"
    decay_rate: float = 0.01
    last_reaffirmed_at: str = ""
    source_utterance_ids: list[str] = field(default_factory=list)
    source_content_ids: list[str] = field(default_factory=list)


@dataclass
class EdgePayload:
    edge_id: str
    from_node: str
    to_node: str
    relation_type: str  # 'supersedes', 'reinforces', 'contradicts', 'elaborates'
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class ChangeLogPayload:
    log_id: str
    entity_type: str
    entity_id: str
    field: str
    old_value: str | None
    new_value: str | None
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class KnowledgePayload:
    """Canonical Contract 2: AI Layer ➔ Database Layer."""

    session: dict[str, Any]
    utterances: list[UtterancePayload] = field(default_factory=list)
    content_units: list[ContentUnitPayload] = field(default_factory=list)
    edges: list[EdgePayload] = field(default_factory=list)
    change_logs: list[ChangeLogPayload] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "utterances": [
                {
                    "utterance_id": u.utterance_id,
                    "session_id": u.session_id,
                    "speaker_id": u.speaker_id,
                    "start_ts": u.start_ts,
                    "end_ts": u.end_ts,
                    "raw_text": u.raw_text,
                    "identity_evidence": u.identity_evidence,
                }
                for u in self.utterances
            ],
            "content_units": [
                {
                    "content_id": c.content_id,
                    "workspace_id": c.workspace_id,
                    "project_id": c.project_id,
                    "type": c.type,
                    "content": c.content,
                    "embedding": c.embedding,
                    "status": c.status,
                    "decay_rate": c.decay_rate,
                    "last_reaffirmed_at": c.last_reaffirmed_at,
                    "source_utterance_ids": c.source_utterance_ids,
                    "source_content_ids": c.source_content_ids,
                }
                for c in self.content_units
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "relation_type": e.relation_type,
                    "confidence": e.confidence,
                    "reasoning": e.reasoning,
                }
                for e in self.edges
            ],
            "change_logs": [
                {
                    "log_id": l.log_id,
                    "entity_type": l.entity_type,
                    "entity_id": l.entity_id,
                    "field": l.field,
                    "old_value": l.old_value,
                    "new_value": l.new_value,
                    "confidence": l.confidence,
                    "reasoning": l.reasoning,
                }
                for l in self.change_logs
            ],
        }
