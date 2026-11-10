"""Native integration is tested without executable downloads or external services."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest

from dsp.diarization.adapters.nemo_native_backend import NemoNativeSpeakerBackend, parse_native_segments
from dsp.diarization.config import LocalDiarizationConfig
from dsp.diarization.native_config import NativeDiarizationConfig
from dsp.exceptions import DiarizationConfigurationError, DiarizationInferenceError
from dsp.pipeline import DSPConfig, DSPPipeline


@pytest.fixture
def config(tmp_path: Path) -> NativeDiarizationConfig:
    runtime = tmp_path / "runtime.exe"
    weights = tmp_path / "model.gguf"
    runtime.write_bytes(b"fixture only; never executed")
    weights.write_bytes(b"mock weights")
    return NativeDiarizationConfig.from_mapping({"executable": str(runtime), "model": str(weights)})


def test_json_preserves_overlap_and_zero_based_slots() -> None:
    rows = parse_native_segments(json.dumps({"segments": [
        {"start": 0, "end": 2, "speaker": 1}, {"start": 1, "end": 3, "speaker": 4}]}))
    assert [(r.start, r.end, r.cluster) for r in rows] == [(0, 2, 0), (1, 3, 3)]


@pytest.mark.parametrize("text", ["bad json", "[]", "{}", '{"segments": null}',
    '{"segments": [null]}', '{"segments": [{"start": 0}]}',
    '{"segments": [{"start": -1, "end": 2, "speaker": 1}]}',
    '{"segments": [{"start": 1, "end": 1, "speaker": 1}]}',
    '{"segments": [{"start": NaN, "end": 2, "speaker": 1}]}',
    '{"segments": [{"start": true, "end": 2, "speaker": 1}]}',
    '{"segments": [{"start": 0, "end": 2, "speaker": true}]}',
    '{"segments": [{"start": 0, "end": 2, "speaker": 9}]}'])
def test_invalid_native_output_is_domain_error(text: str) -> None:
    with pytest.raises(DiarizationInferenceError):
        parse_native_segments(text)


@pytest.mark.parametrize("values", [{"preset": "unknown"}, {"timeout_seconds": 0},
    {"timeout_seconds": float("nan")}, {"onset": 1}, {"offset": -0.1},
    {"pad_onset_seconds": float("inf")}, {"full_attention": "false"},
    {"onset": 0.3, "offset": 0.5}, {"backend": "unknown"},
    {"backend": "nemo_native", "num_speakers": 4}, {"native": {"unknown": 1}}])
def test_bad_config_fails_at_boundary(values: dict[str, object]) -> None:
    if set(values) & {"backend", "native"}:
        with pytest.raises(DiarizationConfigurationError):
            LocalDiarizationConfig.from_mapping(values)
    else:
        with pytest.raises(DiarizationConfigurationError):
            NativeDiarizationConfig.from_mapping(values)


def test_mock_command_pipeline_and_temporary_cleanup(config: NativeDiarizationConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[Path] = []

    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        source = Path(command[2])
        observed.append(source)
        with wave.open(str(source), "rb") as clip:
            assert (clip.getnchannels(), clip.getframerate(), clip.getsampwidth()) == (1, 16000, 2)
            assert clip.getnframes() == 32000
        assert command[command.index("--model") + 1] == str(Path(config.model).resolve())
        assert options["shell"] is False and options["timeout"] == 600.0
        return subprocess.CompletedProcess(command, 0, json.dumps({"segments": [
            {"start": 0, "end": 1.5, "speaker": 3}, {"start": 1, "end": 2, "speaker": 7}]}), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as recording:
        recording.setparams((1, 2, 16000, 32000, "NONE", "not compressed"))
        recording.writeframes((np.sin(np.arange(32000) * 0.1) * 10000).astype("<i2").tobytes())
    pipeline = DSPPipeline(DSPConfig(random_audio_provider="local_model", local_model=LocalDiarizationConfig(backend="nemo_native", native=config)))
    payload = pipeline.process_audio_file(source, tmp_path / "outputs", "mock_native").payload
    assert [r.preliminary_speaker_id for r in payload.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(r.is_overlap and r.speaker_source == "diarization" and Path(r.clip_path).exists() for r in payload.segments)
    assert observed and not observed[0].exists()


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("mock", 1), OSError("runtime unavailable"),
    subprocess.CalledProcessError(1, "mock", stderr="bad model")])
def test_process_errors_are_typed_and_cleaned(config: NativeDiarizationConfig, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    observed: list[Path] = []

    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        observed.append(Path(command[2]))
        raise failure

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(DiarizationInferenceError):
        NemoNativeSpeakerBackend(config).infer(np.ones(16000, dtype=np.float32), 16000)
    assert observed and not observed[0].exists()


def test_missing_assets_and_full_attention_limit(config: NativeDiarizationConfig) -> None:
    with pytest.raises(DiarizationConfigurationError, match="Missing native"):
        replace(config, model="nonexistent-fixture.gguf").validate_assets()
    backend = NemoNativeSpeakerBackend(replace(config, full_attention=True, max_full_attention_seconds=0.5))
    with pytest.raises(DiarizationInferenceError, match="limit exceeded"):
        backend.infer(np.ones(16000, dtype=np.float32), 16000)
    with pytest.raises(DiarizationInferenceError, match="finite 16 kHz"):
        backend.infer(np.array([float("nan")]), 16000)
    with pytest.raises(DiarizationInferenceError, match="finite 16 kHz"):
        backend.infer(np.array([], dtype=np.float32), 16000)
