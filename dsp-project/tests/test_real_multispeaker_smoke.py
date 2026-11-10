"""Opt-in downloaded-model integration; ordinary tests never require these assets."""

from dataclasses import replace
import os
import json
from pathlib import Path
import shutil
import wave
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_REAL_DSP") != "1", reason="Opt-in real model benchmark; no downloads in tests")


@pytest.mark.parametrize("backend", ["sherpa", "nemo_native"])
def test_real_audio_dsp_stt_ai_contract(tmp_path: Path, backend: str) -> None:
    from ai_layer.common.config_loader import load_config
    from ai_layer.common.local_stt import LocalSTTConfig, LocalWhisperSTTProvider
    from ai_layer.pipeline_orchestrator import AILayerOrchestrator
    from dsp import DSPConfig, DSPPipeline
    from dsp.diarization.config import LocalDiarizationConfig
    result = DSPPipeline(DSPConfig(random_audio_provider="local_model", local_model=LocalDiarizationConfig(backend=backend))).process_audio_file(
        "data/benchmarks/1-two-speakers-en.wav", tmp_path, "real_smoke")
    assert {s.preliminary_speaker_id for s in result.payload.segments} == {"SPEAKER_00", "SPEAKER_01"}
    for segment in result.payload.segments:
        with wave.open(segment.clip_path) as clip:
            assert (clip.getframerate(), clip.getnchannels()) == (16000, 1)
            assert abs(clip.getnframes() / 16000 - (segment.end_time - segment.start_time)) < 2 / 16000
    config = load_config("configs/ai_layer_config.yaml")
    config = replace(config, providers=replace(config.providers, mode="mock"))
    completed = AILayerOrchestrator(config, stt_provider=LocalWhisperSTTProvider(LocalSTTConfig())).process_dsp_payload(result.payload)
    assert len(completed.verified_transcript.turns) >= 2
    assert all(t.speaker_name in {"SPEAKER_00", "SPEAKER_01"} for t in completed.verified_transcript.turns)
    assert all("Transcribed speech from" not in t.text for t in completed.verified_transcript.turns)
    assert sum(len(t.text.split()) for t in completed.verified_transcript.turns) > 10


def test_native_runtime_allowlisted_setup_is_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the documented minimal extraction, not a manually unpacked archive."""
    import scripts.setup_diarization as setup
    from dsp.diarization.adapters.nemo_native_backend import NemoNativeSpeakerBackend
    from dsp.diarization.native_config import NativeDiarizationConfig
    from dsp.ingestion.audio_loader import load_wav
    manifest = json.loads(Path("configs/native_diarization_assets.json").read_text(encoding="utf-8"))
    asset = next(entry for entry in manifest["assets"] if "extract_zip_members" in entry)
    archive = Path(asset["path"])
    target = tmp_path / asset["path"]
    target.parent.mkdir(parents=True)
    shutil.copyfile(archive, target)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    setup.prepare_asset(asset, float(manifest["timeout_seconds"]))
    executable = target.parent / "nemo-speech-0.2.0-windows-x86_64-cpu/bin/nemo-speech.exe"
    config = NativeDiarizationConfig.from_mapping({"executable": str(executable),
        "model": str(Path("models/diarization/Nemotron-3-Diarization.q8_0.gguf").resolve())})
    audio = load_wav(Path("data/benchmarks/1-two-speakers-en.wav"))
    result = NemoNativeSpeakerBackend(config).infer(audio.samples[:, 0], audio.metadata.sample_rate)
    assert len({row.cluster for row in result}) == 2


def test_native_four_speaker_reference_and_clips(tmp_path: Path) -> None:
    """Opt-in sample-specific regression, not a general accuracy guarantee."""
    from dsp import DSPConfig, DSPPipeline
    from dsp.diarization.config import LocalDiarizationConfig
    from dsp.evaluation import TimedSpeaker, diarization_error, read_rttm
    result = DSPPipeline(DSPConfig(random_audio_provider="local_model", local_model=LocalDiarizationConfig(backend="nemo_native"))).process_audio_file(
        "data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.wav", tmp_path, "four_speaker_native")
    assert len({s.preliminary_speaker_id for s in result.payload.segments}) == 4
    assert any(s.is_overlap for s in result.payload.segments)
    assert all(s.speaker_source == "diarization" for s in result.payload.segments)
    reference = read_rttm(Path("data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.rttm"))
    prediction = [TimedSpeaker(s.start_time, s.end_time, s.preliminary_speaker_id) for s in result.payload.segments]
    assert diarization_error(reference, prediction)["der"] < 0.23
    for segment in result.payload.segments:
        with wave.open(segment.clip_path) as clip:
            assert clip.getframerate() == 16000 and clip.getnchannels() == 1
            assert abs(clip.getnframes() / 16000 - (segment.end_time - segment.start_time)) < 2 / 16000
