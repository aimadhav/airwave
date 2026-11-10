"""Local speaker logic is verifiable without model downloads or ONNX installed."""

from pathlib import Path
import numpy as np
import pytest
from dsp.contracts import AudioBuffer, AudioMetadata
from dsp.diarization.adapters.mock_backend import MockSpeakerBackend
from dsp.diarization.config import LocalDiarizationConfig
from dsp.diarization.interfaces.inference import ModelSpeakerInterval
from dsp.diarization.services.local_provider import LocalModelDiarizer
from dsp.exceptions import DiarizationConfigurationError, DiarizationInferenceError
from dsp.timeline import TimelineBuilder


def audio(seconds: float = 2.0) -> AudioBuffer:
    frames = int(seconds * 16000)
    return AudioBuffer(np.full((frames, 1), 0.1, dtype=np.float32), AudioMetadata(Path("mock.wav"), 16000, 1, 2, frames))


def test_real_provider_contract_without_real_models() -> None:
    provider = LocalModelDiarizer(MockSpeakerBackend([
        ModelSpeakerInterval(1.0, 2.5, 42), ModelSpeakerInterval(0.2, 1.2, 9),
        ModelSpeakerInterval(1.8, 2.0, 9),
    ]))
    # Empty energy VAD does not incorrectly suppress model-detected speech.
    result = provider.diarize(audio(), [])
    assert [s.speaker_id for s in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert result[1].end_time == 2.0
    assert all(s.confidence == 0 and s.speaker_source == "diarization" for s in result)
    timeline, _ = TimelineBuilder().validate(result, 2.0)
    assert all(s.is_overlap for s in timeline)


@pytest.mark.parametrize("row", [ModelSpeakerInterval(-1, 1, 0), ModelSpeakerInterval(1, 1, 0),
                                ModelSpeakerInterval(0, float("nan"), 0), ModelSpeakerInterval(0, 1, -1)])
def test_invalid_model_intervals_are_typed_errors(row: ModelSpeakerInterval) -> None:
    with pytest.raises(DiarizationInferenceError):
        LocalModelDiarizer(MockSpeakerBackend([row])).diarize(audio(), [])


def test_empty_and_silent_input_is_not_forced_into_a_speaker() -> None:
    provider = LocalModelDiarizer(MockSpeakerBackend([ModelSpeakerInterval(0, 1, 0)]))
    assert provider.diarize(audio(0), []) == []
    silent = audio()
    silent.samples[:] = 0
    assert provider.diarize(silent, []) == []


@pytest.mark.parametrize("values", [{"num_speakers": 0}, {"num_threads": 0}, {"cluster_threshold": float("nan")},
                                   {"window_shift_ratio": 0}, {"min_duration_on": -1}, {"oops": True},
                                   {"num_speakers": 2.5}, {"num_speakers": True}, {"num_threads": 1.2}])
def test_configuration_fails_fast(values: dict[str, object]) -> None:
    with pytest.raises(DiarizationConfigurationError):
        LocalDiarizationConfig.from_mapping(values)


def test_missing_models_give_setup_instructions(tmp_path: Path) -> None:
    with pytest.raises(DiarizationConfigurationError, match="setup_diarization"):
        LocalDiarizationConfig(segmentation_model=str(tmp_path / "missing.onnx")).validate_assets()
