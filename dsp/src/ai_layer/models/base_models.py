from dataclasses import dataclass, field


@dataclass
class TimeRange:
    start_time_seconds: float
    end_time_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time_seconds - self.start_time_seconds)


@dataclass
class SessionMetadata:
    session_id: str
    session_title: str
    recorded_at: str
    participant_ids: list[str] = field(default_factory=list)
    project_id: str | None = None
