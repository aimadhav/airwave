from ai_layer.common.config_loader import load_config
from ai_layer.models.base_models import SessionMetadata, TimeRange
from ai_layer.models.speaker_models import SpeakerProfile
from ai_layer.models.transcription_models import AudioClip
from ai_layer.pipeline_orchestrator import AILayerOrchestrator


def sample_clips() -> list[AudioClip]:
    return [
        AudioClip(
            clip_id="clip_1",
            audio_path="clips/clip_1.wav",
            preliminary_speaker_id="Speaker_1",
            time_range=TimeRange(0.0, 2.5),
        ),
        AudioClip(
            clip_id="clip_2",
            audio_path="clips/clip_2.wav",
            preliminary_speaker_id="Speaker_2",
            time_range=TimeRange(3.0, 6.0),
        ),
        AudioClip(
            clip_id="clip_3",
            audio_path="clips/clip_3.wav",
            preliminary_speaker_id="Speaker_2",
            time_range=TimeRange(6.2, 9.0),
        ),
        AudioClip(
            clip_id="clip_4",
            audio_path="clips/clip_4.wav",
            preliminary_speaker_id="Speaker_1",
            time_range=TimeRange(10.0, 14.0),
        ),
        AudioClip(
            clip_id="clip_5",
            audio_path="clips/clip_5.wav",
            preliminary_speaker_id="Speaker_3",
            time_range=TimeRange(15.0, 19.0),
        ),
    ]


def sample_session_metadata() -> SessionMetadata:
    return SessionMetadata(
        session_id="session_001",
        session_title="Product Launch Planning",
        recorded_at="2026-09-17T10:00:00",
        participant_ids=["Speaker_1", "Speaker_2", "Speaker_3"],
        project_id="project_launch",
    )


def sample_speakers() -> list[SpeakerProfile]:
    return [
        SpeakerProfile(
            speaker_id="speaker_sarah",
            canonical_name="Sarah",
            aliases=["Sara"],
            base_confidence=0.6,
        )
    ]


def orchestrator() -> AILayerOrchestrator:
    return AILayerOrchestrator.from_config_file("configs/ai_layer_config.yaml")
