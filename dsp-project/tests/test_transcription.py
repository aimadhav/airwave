from ai_layer.common.config_loader import load_config
from ai_layer.common.stt_provider import MockSTTProvider
from ai_layer.models.base_models import TimeRange
from ai_layer.models.transcription_models import AudioClip
from ai_layer.transcription.audio_transcriber import AudioTranscriber
from ai_layer.transcription.clip_sorter import ClipSorter
from ai_layer.transcription.service import STTTranscriptionService
from ai_layer.transcription.turn_stitcher import TurnStitcher
from tests.conftest import sample_clips


def test_clip_sorter():
    clips = [
        AudioClip("c2", "b.wav", "Speaker_2", TimeRange(5.0, 7.0)),
        AudioClip("c1", "a.wav", "Speaker_1", TimeRange(0.0, 2.0)),
    ]
    sorted_clips = ClipSorter.sort_by_start_time(clips)
    ClipSorter.validate_timestamps(sorted_clips)
    assert [clip.clip_id for clip in sorted_clips] == ["c1", "c2"]


def test_audio_transcriber():
    transcriber = AudioTranscriber(MockSTTProvider())
    utterances = transcriber.transcribe_batch(sample_clips())
    assert len(utterances) == 5
    assert utterances[0].text.startswith("Hello everyone")


def test_turn_stitcher():
    transcriber = AudioTranscriber(MockSTTProvider())
    utterances = transcriber.transcribe_batch(sample_clips())
    stitcher = TurnStitcher(max_pause_seconds=1.5)
    turns = stitcher.stitch_consecutive_turns(utterances)
    assert len(turns) == 4
    assert any("Sarah" in turn.text for turn in turns)


def test_transcription_service():
    config = load_config("configs/ai_layer_config.yaml")
    service = STTTranscriptionService(MockSTTProvider(), config.transcription)
    output = service.process_session_clips("session_001", sample_clips())
    assert output.session_id == "session_001"
    assert len(output.raw_utterances) == 5
    assert len(output.turns) >= 4
