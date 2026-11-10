from dataclasses import dataclass, field


@dataclass
class SessionSummary:
    session_id: str
    title: str
    overview: str
    discussion_highlights: list[str] = field(default_factory=list)
    decisions_summary: list[str] = field(default_factory=list)
    action_items_summary: list[str] = field(default_factory=list)


@dataclass
class ProjectSummary:
    project_id: str
    title: str
    status_overview: str
    latest_milestones: list[str] = field(default_factory=list)
    open_action_items: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
