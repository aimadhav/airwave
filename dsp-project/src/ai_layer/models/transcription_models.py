from dataclasses import dataclass, field
from typing import Literal

from ai_layer.models.base_models import TimeRange


@dataclass
class AudioClip:
    clip_id: str
    audio_path: str
    preliminary_speaker_id: str
    time_range: TimeRange


@dataclass
class RawUtterance:
    utterance_id: str
    speaker_id: str
    text: str
    time_range: TimeRange
    audio_path: str


@dataclass
class StitchedTurn:
    turn_id: str
    speaker_id: str
    text: str
    time_range: TimeRange
    source_utterance_ids: list[str] = field(default_factory=list)


@dataclass
class STTTranscriptOutput:
    session_id: str
    turns: list[StitchedTurn] = field(default_factory=list)
    raw_utterances: list[RawUtterance] = field(default_factory=list)


@dataclass
class AudioClipSegment:
    """Acoustic segment produced by the DSP layer conforming to Contract 1."""

    segment_id: str
    clip_path: str
    start_time: float
    end_time: float
    preliminary_speaker_id: str = "SPK_UNKNOWN"
    acoustic_confidence: float = 1.0
    is_overlap_separated: bool = False
    voice_embedding: list[float] = field(default_factory=list)
    speaker_source: Literal[
        "meet_transcript", "diarization", "speaker_match", "unknown"
    ] = "unknown"
    is_overlap: bool = False


@dataclass
class AudioClipPayload:
    """Canonical Contract 1: DSP Layer ➔ AI Layer."""

    session_id: str
    source_audio_path: str
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: float = 0.0
    segments: list[AudioClipSegment] = field(default_factory=list)
    source_type: Literal["google_meet", "audio_file"] = "audio_file"
    speaker_source: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_audio_clips(self) -> list[AudioClip]:
        """Converts Contract 1 segments to internal AudioClip instances."""
        return [
            AudioClip(
                clip_id=seg.segment_id,
                audio_path=seg.clip_path,
                preliminary_speaker_id=seg.preliminary_speaker_id,
                time_range=TimeRange(
                    start_time_seconds=seg.start_time,
                    end_time_seconds=seg.end_time,
                ),
            )
            for seg in self.segments
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "AudioClipPayload":
        segments = [
            AudioClipSegment(
                segment_id=s.get("segment_id", f"seg_{i:03d}"),
                clip_path=s.get("clip_path", ""),
                start_time=float(s.get("start_time", 0.0)),
                end_time=float(s.get("end_time", 0.0)),
                preliminary_speaker_id=s.get("preliminary_speaker_id", "SPK_UNKNOWN"),
                acoustic_confidence=float(s.get("acoustic_confidence", 1.0)),
                is_overlap_separated=bool(s.get("is_overlap_separated", False)),
                voice_embedding=s.get("voice_embedding", []),
                speaker_source=s.get("speaker_source", "unknown"),
                is_overlap=bool(s.get("is_overlap", False)),
            )
            for i, s in enumerate(data.get("segments", []))
        ]
        return cls(
            session_id=data.get("session_id", ""),
            source_audio_path=data.get("source_audio_path", ""),
            sample_rate=int(data.get("sample_rate", 16000)),
            channels=int(data.get("channels", 1)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            segments=segments,
            source_type=data.get("source_type", "audio_file"),
            speaker_source=data.get("speaker_source", "unknown"),
            warnings=list(data.get("warnings", [])),
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source_audio_path": self.source_audio_path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "source_type": self.source_type,
            "speaker_source": self.speaker_source,
            "warnings": list(self.warnings),
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "clip_path": s.clip_path,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "preliminary_speaker_id": s.preliminary_speaker_id,
                    "acoustic_confidence": s.acoustic_confidence,
                    "is_overlap_separated": s.is_overlap_separated,
                    "voice_embedding": s.voice_embedding,
                    "speaker_source": s.speaker_source,
                    "is_overlap": s.is_overlap,
                }
                for s in self.segments
            ],
        }
