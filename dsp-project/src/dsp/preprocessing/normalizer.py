"""Canonical audio conditioning for the DSP front end."""

from dataclasses import replace
from math import gcd

import numpy as np

from dsp.contracts import AudioBuffer, AudioMetadata
from dsp.exceptions import DiarizationConfigurationError


def resample_antialiased(audio: AudioBuffer, target_sample_rate: int) -> AudioBuffer:
    """Polyphase low-pass resampling for learned voice models, avoiding aliasing."""
    if target_sample_rate <= 0:
        raise DiarizationConfigurationError("Sample rate must be positive")
    if audio.metadata.sample_rate == target_sample_rate:
        return audio
    try:
        from scipy.signal import resample_poly
    except ImportError as error:
        raise DiarizationConfigurationError("Local model resampling requires scipy; install requirements-diarization.txt") from error
    divisor = gcd(audio.metadata.sample_rate, target_sample_rate)
    if audio.samples.size:
        samples = resample_poly(audio.samples, target_sample_rate // divisor,
                                audio.metadata.sample_rate // divisor, axis=0).astype(np.float32)
    else:
        samples = audio.samples.copy()
    return AudioBuffer(samples, replace(audio.metadata, sample_rate=target_sample_rate, frame_count=len(samples)))


def to_mono(audio: AudioBuffer) -> AudioBuffer:
    """Average channels into one channel while preserving the duration."""

    if audio.metadata.channels == 1:
        return audio
    samples = audio.samples.mean(axis=1, keepdims=True).astype(np.float32)
    metadata = replace(audio.metadata, channels=1)
    return AudioBuffer(samples=samples, metadata=metadata)


def normalize_peak(audio: AudioBuffer, target_peak: float = 0.95) -> AudioBuffer:
    """Remove DC offset and scale the signal to a safe peak level."""

    if not 0 < target_peak <= 1:
        raise ValueError("target_peak must be in the range (0, 1]")
    samples = audio.samples.astype(np.float32, copy=True)
    if samples.size == 0:
        return AudioBuffer(samples=samples, metadata=audio.metadata)
    samples -= samples.mean(axis=0, keepdims=True)
    peak = float(np.max(np.abs(samples), initial=0.0))
    if peak > 0:
        samples *= target_peak / peak
    return AudioBuffer(samples=samples, metadata=audio.metadata)


def resample_linear(audio: AudioBuffer, target_sample_rate: int) -> AudioBuffer:
    """Resample using linear interpolation for the dependency-light baseline."""

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    source_rate = audio.metadata.sample_rate
    if source_rate == target_sample_rate:
        return audio
    source_count = audio.samples.shape[0]
    if source_count == 0:
        return AudioBuffer(audio.samples.copy(), replace(audio.metadata, sample_rate=target_sample_rate))
    target_count = max(1, round(source_count * target_sample_rate / source_rate))
    source_positions = np.arange(source_count, dtype=np.float64)
    target_positions = np.linspace(0, source_count - 1, target_count)
    channels = [
        np.interp(target_positions, source_positions, audio.samples[:, channel])
        for channel in range(audio.metadata.channels)
    ]
    samples = np.stack(channels, axis=1).astype(np.float32)
    metadata = AudioMetadata(
        source_path=audio.metadata.source_path,
        sample_rate=target_sample_rate,
        channels=audio.metadata.channels,
        sample_width=audio.metadata.sample_width,
        frame_count=target_count,
    )
    return AudioBuffer(samples=samples, metadata=metadata)
