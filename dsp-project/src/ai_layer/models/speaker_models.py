from dataclasses import dataclass, field

from ai_layer.models.base_models import TimeRange


@dataclass
class SpeakerProfile:
    speaker_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    base_confidence: float = 0.5


@dataclass
class IdentityCue:
    cue_type: str  # 'self_intro', 'direct_address', 'third_party'
    detected_name: str
    target_speaker_id: str
    source_turn_id: str
    confidence: float = 0.8


@dataclass
class SpeakerReassignment:
    preliminary_speaker_id: str
    resolved_speaker_id: str
    resolved_name: str
    rationale: str = ""


@dataclass
class VerifiedTurn:
    turn_id: str
    speaker_id: str
    speaker_name: str
    text: str
    time_range: TimeRange


@dataclass
class VerifiedTranscript:
    session_id: str
    turns: list[VerifiedTurn] = field(default_factory=list)
    reassignments: list[SpeakerReassignment] = field(default_factory=list)
