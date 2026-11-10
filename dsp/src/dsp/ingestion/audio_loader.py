"""Dependency-light WAV ingestion for the first DSP milestone."""

from pathlib import Path
import wave

import numpy as np

from dsp.contracts import AudioBuffer, AudioMetadata
from dsp.exceptions import AudioLoadError, UnsupportedAudioError


def load_wav(path: str | Path) -> AudioBuffer:
    """Load a PCM WAV file into a float32 ``(frames, channels)`` buffer.

    The first milestone intentionally supports PCM WAV only. MP3/M4A/FLAC
    adapters can be added later without changing the ``AudioBuffer`` contract.
    """

    source_path = Path(path)
    try:
        with wave.open(str(source_path), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            raw = reader.readframes(frame_count)
            sample_format = reader.getcomptype()
    except (FileNotFoundError, OSError, wave.Error) as exc:
        raise AudioLoadError(f"Could not load WAV file: {source_path}") from exc

    if sample_format != "NONE":
        raise UnsupportedAudioError(
            f"Compressed WAV is not supported: {source_path} ({sample_format})"
        )
    if sample_width not in (1, 2, 4):
        raise UnsupportedAudioError(
            f"Only 8-, 16-, and 32-bit PCM WAV files are supported, got {sample_width * 8} bits"
        )

    samples = _pcm_bytes_to_float(raw, sample_width)
    if samples.size != frame_count * channels:
        raise AudioLoadError(f"WAV frame data is truncated: {source_path}")
    samples = samples.reshape(frame_count, channels)

    metadata = AudioMetadata(
        source_path=source_path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frame_count=frame_count,
    )
    return AudioBuffer(samples=samples, metadata=metadata)


def _pcm_bytes_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        # PCM WAV stores 8-bit samples as unsigned values.
        return ((np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0)
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
