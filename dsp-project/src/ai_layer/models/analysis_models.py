from dataclasses import dataclass, field

from ai_layer.models.base_models import TimeRange


@dataclass
class UtteranceCitation:
    turn_id: str
    speaker_name: str
    time_range: TimeRange
    spoken_text: str


@dataclass
class ExtractedFact:
    fact_id: str
    statement: str
    citation: UtteranceCitation


@dataclass
class ExtractedDecision:
    decision_id: str
    decision: str
    agreed_by: list[str] = field(default_factory=list)
    citation: UtteranceCitation | None = None


@dataclass
class ActionItem:
    task_id: str
    task_description: str
    assignee_name: str
    deadline: str | None = None
    citation: UtteranceCitation | None = None


@dataclass
class AnalysisReport:
    session_id: str
    facts: list[ExtractedFact] = field(default_factory=list)
    decisions: list[ExtractedDecision] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
