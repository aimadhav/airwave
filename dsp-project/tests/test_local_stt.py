"""The optional STT boundary works with an injected runtime, without downloads."""

from pathlib import Path
from types import SimpleNamespace
import wave
import numpy as np
import pytest
import ai_layer.common.local_stt as module
from ai_layer.models.base_models import TimeRange
from ai_layer.models.transcription_models import AudioClip


def test_real_stt_contract_with_fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assets = [tmp_path / name for name in ("encoder", "decoder", "tokens")]
    for path in assets:
        path.write_bytes(b"fixture")
    received: list[tuple[int, int]] = []
    class Stream:
        result = SimpleNamespace(text="  Real test words.  ")
        def accept_waveform(self, rate: int, samples: np.ndarray) -> None:
            received.append((rate, len(samples)))
    engine = SimpleNamespace(create_stream=Stream, decode_stream=lambda stream: None)
    runtime = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_whisper=lambda **kwargs: engine))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: runtime)
    provider = module.LocalWhisperSTTProvider(module.LocalSTTConfig(*(str(path) for path in assets)))
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.ones(16000, dtype="<i2").tobytes())
    clip = AudioClip("a", str(audio), "SPEAKER_00", TimeRange(0, 1))
    assert provider.transcribe_clip(clip) == "Real test words."
    assert received == [(16000, 16000)]
    provider._config = module.LocalSTTConfig(*(str(path) for path in assets), max_chunk_seconds=0.4)
    assert provider.transcribe_clip(clip) == "Real test words. Real test words. Real test words."
    assert received[1:] == [(16000, 6400), (16000, 6400), (16000, 3200)]
    whole_file = provider.transcribe_file_segments(str(audio))
    assert whole_file[0]["speaker_id"] == "SPK_UNKNOWN"
    assert whole_file[0]["end"] == 1.0
    clip.audio_path = str(tmp_path / "missing.wav")
    with pytest.raises(module.LocalSTTError, match="Failed to transcribe"):
        provider.transcribe_clip(clip)


def test_missing_assets_fail_with_setup_hint(tmp_path: Path) -> None:
    with pytest.raises(module.LocalSTTError, match="setup_diarization"):
        module.LocalWhisperSTTProvider(module.LocalSTTConfig(encoder=str(tmp_path / "missing")))
