"""Tests for the deterministic voice activity detector."""

import numpy as np

from dsp.contracts import AudioBuffer, AudioMetadata
from dsp.vad.energy_vad import EnergyVAD


def _audio(samples: np.ndarray, sample_rate: int = 1_000) -> AudioBuffer:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
    return AudioBuffer(
        samples=samples,
        metadata=AudioMetadata(
            source_path="fixture.wav",
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            frame_count=len(samples),
        ),
    )


def test_vad_returns_no_segments_for_silence() -> None:
    result = EnergyVAD().detect(_audio(np.zeros(2_000)))

    assert result == []


def test_vad_detects_a_speech_region() -> None:
    samples = np.zeros(2_000, dtype=np.float32)
    samples[500:1_200] = 0.4

    result = EnergyVAD(min_speech_ms=100).detect(_audio(samples))

    assert len(result) == 1
    assert 0.45 <= result[0].start_time <= 0.51
    assert 1.19 <= result[0].end_time <= 1.25
    assert 0.0 <= result[0].confidence <= 1.0


def test_vad_merges_short_silence_and_keeps_long_silence_separate() -> None:
    samples = np.zeros(4_000, dtype=np.float32)
    samples[500:1_000] = 0.4
    samples[1_050:1_500] = 0.4
    samples[2_000:2_800] = 0.4

    result = EnergyVAD(min_speech_ms=100, min_silence_ms=100).detect(_audio(samples))

    assert len(result) == 2
    assert result[0].end_time < result[1].start_time


def test_vad_adapts_to_a_raised_noise_floor_without_lowering_baseline() -> None:
    vad = EnergyVAD(threshold_db=-35.0, noise_margin_db=3.0)

    db = np.array(
        [-40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -40.0, -20.0],
        dtype=np.float32,
    )

    assert vad._effective_threshold(db) == -35.0
    assert EnergyVAD(threshold_db=-35.0, noise_margin_db=10.0)._effective_threshold(db) == -30.0
