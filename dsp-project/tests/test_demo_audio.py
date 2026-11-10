"""Playback copies preserve recording timing and never overwrite originals."""

from pathlib import Path
import wave

import numpy as np
import pytest
from scripts.normalize_demo_audio import normalize_copy


def test_louder_copy_preserves_frames_and_original(tmp_path: Path) -> None:
    source, output = tmp_path / "original.wav", tmp_path / "louder.wav"
    samples = (np.sin(np.arange(1600) * 0.1) * 3000).astype("<i2")
    with wave.open(str(source), "wb") as recording:
        recording.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
        recording.writeframes(samples.tobytes())
    before = source.read_bytes()
    report = normalize_copy(source, output, 0.95)
    assert report["duration_seconds"] == 0.1 and source.read_bytes() == before
    with wave.open(str(output), "rb") as recording:
        assert recording.getnframes() == 1600
        pcm = np.frombuffer(recording.readframes(1600), dtype="<i2")
        assert max(abs(pcm)) / 32767 == pytest.approx(0.95, abs=1/32767)
    with pytest.raises(FileExistsError):
        normalize_copy(source, output, 0.95)


@pytest.mark.parametrize("peak", [0, -1, 2, float("nan")])
def test_invalid_playback_peak_fails_fast(tmp_path: Path, peak: float) -> None:
    with pytest.raises(ValueError):
        normalize_copy(tmp_path / "unused.wav", tmp_path / "output.wav", peak)
