from pathlib import Path
import numpy as np
import pytest
from dsp.contracts import AudioBuffer, AudioMetadata
from dsp.preprocessing.normalizer import normalize_peak, resample_antialiased

pytest.importorskip("scipy")


def test_low_pass_resampling_rejects_above_nyquist() -> None:
    time = np.arange(48000) / 48000
    samples = np.sin(2 * np.pi * 12000 * time).astype(np.float32)[:, None]
    original = AudioBuffer(samples, AudioMetadata(Path("fixture"), 48000, 1, 2, len(samples)))
    result = resample_antialiased(original, 16000)
    assert result.samples.shape == (16000, 1)
    assert np.sqrt(np.mean(result.samples[100:-100] ** 2)) < 0.01
    assert result.duration_seconds == original.duration_seconds


def test_empty_normalization_and_resampling_are_finite() -> None:
    audio = AudioBuffer(np.empty((0, 1), dtype=np.float32), AudioMetadata(Path("fixture"), 48000, 1, 2, 0))
    assert normalize_peak(audio).samples.size == 0
    assert resample_antialiased(audio, 16000).metadata.frame_count == 0
