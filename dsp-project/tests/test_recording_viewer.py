"""Word alignment and stacked overlap behavior work without real models."""

from pathlib import Path
from dataclasses import replace
import sys
from types import SimpleNamespace
import wave
import numpy as np
import pytest
from recording_viewer.adapters.faster_whisper import FasterWhisperTranscriber
from recording_viewer.config import RecordingViewerConfig
from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.adapters.mock_adapter import MockTimestampedTranscriber
from recording_viewer.exceptions import TranscriptInputError
from recording_viewer.models import TranscriptWord, VoiceRegion
from recording_viewer.services.alignment import align_words, group_turns, union_duration


def test_overlap_words_are_not_duplicated_on_secondary_speaker() -> None:
    words = [TranscriptWord(0, .5, "Hello"), TranscriptWord(1.1, 1.5, " there.")]
    regions = [VoiceRegion("a", "speaker_a", 0, 2), VoiceRegion("b", "speaker_b", 1, 3)]
    turns = align_words(words, regions, .5)
    assert turns[0].text == "Hello there."
    assert turns[1].text == ""  # Never pretend the dominant text is both people's speech.
    assert all(turn.overlap and turn.uncertain for turn in turns)
    assert sum(turn.word_count for turn in turns) == len(words)
    groups = group_turns(turns)
    assert len(groups) == 1 and groups[0].overlap
    assert groups[0].turn_ids == ("a", "b")


def test_non_overlapping_and_touching_turns_stay_separate() -> None:
    words = [TranscriptWord(0, .5, "Hello"), TranscriptWord(1, 1.5, " Hi")]
    turns = align_words(words, [VoiceRegion("a", "s1", 0, 1), VoiceRegion("b", "s2", 1, 2)], .5)
    assert [turn.text for turn in turns] == ["Hello", "Hi"]
    assert not any(turn.overlap or turn.uncertain for turn in turns)
    assert len(group_turns(turns)) == 2


def test_missing_speaker_stays_unknown_and_word_count_is_conserved() -> None:
    words = MockTimestampedTranscriber([TranscriptWord(2, 3, "Unknown voice")]).transcribe(Path("fixture.wav"))
    result = align_words(words, [], .5)
    assert result[0].speaker == "UNKNOWN" and result[0].uncertain
    assert result[0].text == "Unknown voice" and result[0].word_count == 1


def test_same_speaker_overlaps_do_not_become_two_people() -> None:
    turns = align_words([TranscriptWord(.4, .8, "Test")],
                        [VoiceRegion("a", "s1", 0, 1), VoiceRegion("b", "s1", .3, 1.3)], .5)
    assert not any(turn.overlap for turn in turns)
    assert not group_turns(turns)[0].overlap
    assert sum(turn.word_count for turn in turns) == 1
    assert union_duration([(0, 1), (.5, 1.5), (.7, .8)]) == 1.5


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (0, float("nan")), (0, float("inf"))])
def test_invalid_word_timestamps(start: float, end: float) -> None:
    with pytest.raises(TranscriptInputError):
        TranscriptWord(start, end, "invalid")


def test_low_timing_coverage_is_explicitly_uncertain() -> None:
    result = align_words([TranscriptWord(0, 1, "Hi")], [VoiceRegion("a", "s1", .8, 2)], .5)
    assert result[0].uncertain


def test_duplicate_region_ids_are_rejected() -> None:
    with pytest.raises(TranscriptInputError):
        align_words([], [VoiceRegion("a", "s1", 0, 1), VoiceRegion("a", "s2", 0, 1)], .5)


def test_empty_input() -> None:
    assert align_words([], [], .5) == []
    assert group_turns([]) == []


def test_real_adapter_contract_uses_pcm_and_no_model_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"fixture; not executed")
    source = tmp_path / "recording.wav"
    with wave.open(str(source), "wb") as recording:
        recording.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        recording.writeframes(np.ones(16000, dtype="<i2").tobytes())
    config = replace(RecordingViewerConfig.load(Path("configs/recording_viewer.json")), model_directory=str(model))

    class FakeModel:
        def __init__(self, path: str, **settings: object) -> None:
            assert path == str(model) and settings["local_files_only"] is True

        def transcribe(self, samples: np.ndarray, **settings: object) -> tuple[list[object], None]:
            assert samples.dtype == np.float32 and samples.size == 16000
            assert settings["word_timestamps"] is True and settings["vad_filter"] is False
            return [SimpleNamespace(end=1, words=[SimpleNamespace(start=0, end=1, word="Hello")])], None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    assert FasterWhisperTranscriber(config).transcribe(source) == [TranscriptWord(0, 1, "Hello")]


def test_missing_recognition_model_is_not_implicitly_downloaded(tmp_path: Path) -> None:
    config = replace(RecordingViewerConfig.load(Path("configs/recording_viewer.json")), model_directory=str(tmp_path))
    with pytest.raises(RecordingViewerError, match="will not download"):
        FasterWhisperTranscriber(config)


def test_long_recording_policy_is_per_recording_not_a_global_upload_change() -> None:
    import json
    config = RecordingViewerConfig.load(Path("configs/recording_viewer.json"))
    assert config.dsp_config_path is None
    override = replace(config, dsp_config_path="configs/dsp_full_recording_test.yaml", max_duration_seconds=3600)
    assert override.max_duration_seconds == 3600
    assert json.loads(Path("configs/recording_upload.json").read_text())["max_audio_seconds"] == 1800
    with pytest.raises(RecordingViewerError):
        replace(config, dsp_config_path="")
