"""Write reproducible WAV clips from validated speaker segments."""

from dataclasses import replace
from pathlib import Path
import wave

import numpy as np

from dsp.contracts import AudioBuffer, SpeakerSegment


def write_clips(
    audio: AudioBuffer,
    segments: list[SpeakerSegment],
    output_dir: str | Path,
) -> list[SpeakerSegment]:
    """Write one 16-bit PCM WAV clip per segment and attach its path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[SpeakerSegment] = []
    for index, segment in enumerate(segments):
        start_frame = max(0, round(segment.start_time * audio.metadata.sample_rate))
        end_frame = min(
            audio.metadata.frame_count,
            round(segment.end_time * audio.metadata.sample_rate),
        )
        if end_frame <= start_frame:
            continue
        clip_path = output_dir / f"segment_{index + 1:04d}.wav"
        samples = np.clip(audio.samples[start_frame:end_frame], -1.0, 1.0)
        pcm = (samples * 32767.0).astype("<i2").tobytes()
        with wave.open(str(clip_path), "wb") as writer:
            writer.setnchannels(audio.metadata.channels)
            writer.setsampwidth(2)
            writer.setframerate(audio.metadata.sample_rate)
            writer.writeframes(pcm)
        written.append(replace(segment, clip_path=clip_path))
    return written
