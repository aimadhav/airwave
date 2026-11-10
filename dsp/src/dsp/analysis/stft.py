"""NumPy STFT implementation used by the first DSP milestone."""

import numpy as np

from dsp.contracts import AudioBuffer


def stft(
    audio: AudioBuffer,
    frame_length: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """Return a one-sided complex STFT for mono audio.

    The result has shape ``(frequency_bins, frames)``. The implementation is
    deliberately small and deterministic; a later optimization can replace
    it with a specialized backend without changing the caller contract.
    """

    if audio.metadata.channels != 1:
        raise ValueError("STFT currently requires mono audio")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if hop_length > frame_length:
        raise ValueError("hop_length cannot exceed frame_length")

    signal = audio.samples[:, 0]
    frame_count = max(1, int(np.ceil(max(1, len(signal) - frame_length) / hop_length)) + 1)
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(signal, (0, max(0, padded_length - len(signal))))
    window = np.hanning(frame_length).astype(np.float32)
    frames = np.stack(
        [padded[start : start + frame_length] * window for start in range(0, frame_count * hop_length, hop_length)],
        axis=0,
    )
    return np.fft.rfft(frames, axis=1).T
