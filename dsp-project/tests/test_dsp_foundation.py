"""Tests for the first DSP implementation milestone."""

from pathlib import Path
import wave

import numpy as np

from dsp.analysis.stft import stft
from dsp.ingestion.audio_loader import load_wav
from dsp.preprocessing.normalizer import normalize_peak, resample_linear, to_mono


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 8_000) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    interleaved = (pcm * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(samples.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(interleaved)


def test_load_wav_returns_float_audio_metadata(tmp_path: Path) -> None:
    samples = np.zeros((800, 2), dtype=np.float32)
    samples[100, 0] = 0.5
    path = tmp_path / "input.wav"
    _write_wav(path, samples)

    audio = load_wav(path)

    assert audio.samples.shape == (800, 2)
    assert audio.metadata.sample_rate == 8_000
    assert audio.metadata.channels == 2
    assert audio.duration_seconds == 0.1
    assert np.isclose(audio.samples[100, 0], 0.5, atol=1e-4)


def test_preprocessing_converts_to_mono_resamples_and_normalizes(tmp_path: Path) -> None:
    samples = np.column_stack(
        [np.full(800, 0.2, dtype=np.float32), np.full(800, 0.4, dtype=np.float32)]
    )
    path = tmp_path / "input.wav"
    _write_wav(path, samples)

    processed = normalize_peak(resample_linear(to_mono(load_wav(path)), 16_000))

    assert processed.samples.shape == (1_600, 1)
    assert processed.metadata.sample_rate == 16_000
    assert processed.metadata.channels == 1
    assert np.isclose(float(processed.samples.mean()), 0.0, atol=1e-6)
    assert np.max(np.abs(processed.samples)) == 0.0


def test_stft_returns_expected_frequency_bins(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = np.sin(2 * np.pi * 440 * time).reshape(-1, 1).astype(np.float32)
    path = tmp_path / "tone.wav"
    _write_wav(path, samples, sample_rate)

    audio = to_mono(load_wav(path))
    spectrum = stft(audio, frame_length=256, hop_length=128)

    assert spectrum.shape[0] == 129
    assert spectrum.shape[1] > 1
    assert np.isfinite(spectrum).all()
