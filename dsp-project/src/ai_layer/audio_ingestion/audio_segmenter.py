"""Audio segmentation and Voice Activity Detection (VAD) service.

Slices raw continuous WAV audio recordings into speech segment clips
tagged with preliminary speaker IDs and start/end timestamps.
"""

import math
import struct
import wave
from pathlib import Path

from ai_layer.models.base_models import TimeRange
from ai_layer.models.transcription_models import AudioClip


class AudioSegmenter:
    """Segments raw WAV audio files into discrete speech clips using energy-based VAD."""

    def __init__(
        self,
        frame_duration_ms: int = 50,
        energy_threshold_multiplier: float = 1.3,
        min_speech_duration_sec: float = 0.5,
        max_silence_bridge_sec: float = 0.4,
    ) -> None:
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_silence_bridge_sec = max_silence_bridge_sec

    def segment_wav(
        self,
        wav_path: str | Path,
        output_dir: str | Path,
        session_id: str,
    ) -> list[AudioClip]:
        """Segments a WAV file into speech clips and writes them to output_dir."""
        input_path = Path(wav_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        clips_dir = Path(output_dir) / session_id
        clips_dir.mkdir(parents=True, exist_ok=True)

        with wave.open(str(input_path), "rb") as handle:
            n_channels = handle.getnchannels()
            samp_width = handle.getsampwidth()
            framerate = handle.getframerate()
            n_frames = handle.getnframes()
            raw_data = handle.readframes(n_frames)

        # Convert to mono 16-bit samples
        samples = self._extract_mono_samples(raw_data, n_channels, samp_width)
        total_duration = len(samples) / framerate

        # Compute energy per frame
        frame_size = int(framerate * (self.frame_duration_ms / 1000.0))
        if frame_size <= 0:
            frame_size = 800

        energies: list[float] = []
        for i in range(0, len(samples), frame_size):
            chunk = samples[i : i + frame_size]
            if not chunk:
                continue
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
            energies.append(rms)

        if not energies:
            return []

        avg_energy = sum(energies) / len(energies)
        threshold = max(200.0, avg_energy * self.energy_threshold_multiplier)

        # Identify speech frames
        is_speech = [e > threshold for e in energies]

        # Group speech frames into contiguous intervals
        frame_time = self.frame_duration_ms / 1000.0
        intervals: list[tuple[float, float]] = []

        in_speech = False
        start_time = 0.0
        last_speech_time = 0.0

        for idx, active in enumerate(is_speech):
            current_time = idx * frame_time
            if active:
                if not in_speech:
                    in_speech = True
                    start_time = current_time
                last_speech_time = current_time + frame_time
            else:
                if in_speech:
                    # Check if silence is short enough to bridge
                    silence_duration = current_time - last_speech_time
                    if silence_duration > self.max_silence_bridge_sec:
                        in_speech = False
                        duration = last_speech_time - start_time
                        if duration >= self.min_speech_duration_sec:
                            intervals.append((start_time, min(total_duration, last_speech_time)))

        if in_speech:
            duration = last_speech_time - start_time
            if duration >= self.min_speech_duration_sec:
                intervals.append((start_time, min(total_duration, last_speech_time)))

        # If no intervals detected (e.g. quiet audio), fall back to uniform segments
        if not intervals:
            intervals = [(0.0, total_duration)]

        # Slice and export clips
        clips: list[AudioClip] = []
        speaker_idx = 1

        for i, (start_ts, end_ts) in enumerate(intervals, start=1):
            clip_id = f"clip_{session_id}_{i:03d}"
            clip_filename = f"{clip_id}.wav"
            clip_path = clips_dir / clip_filename

            start_sample = int(start_ts * framerate)
            end_sample = int(end_ts * framerate)
            clip_samples = samples[start_sample:end_sample]

            self._write_mono_wav(clip_path, clip_samples, framerate)

            # Assign alternating preliminary speaker IDs
            preliminary_id = f"Speaker_{speaker_idx}"
            speaker_idx = (speaker_idx % 3) + 1  # Cycle between 1, 2, 3

            clips.append(
                AudioClip(
                    clip_id=clip_id,
                    audio_path=str(clip_path),
                    preliminary_speaker_id=preliminary_id,
                    time_range=TimeRange(
                        start_time_seconds=round(start_ts, 2),
                        end_time_seconds=round(end_ts, 2),
                    ),
                )
            )

        return clips

    @staticmethod
    def _extract_mono_samples(raw_data: bytes, n_channels: int, samp_width: int) -> list[int]:
        if samp_width != 2:
            # Only 16-bit PCM handled directly
            return []
        total_samples = len(raw_data) // 2
        unpacked = struct.unpack(f"<{total_samples}h", raw_data)
        if n_channels == 1:
            return list(unpacked)
        # Average channels to mono
        mono: list[int] = []
        for i in range(0, len(unpacked), n_channels):
            avg = sum(unpacked[i : i + n_channels]) // n_channels
            mono.append(avg)
        return mono

    @staticmethod
    def _write_mono_wav(file_path: Path, samples: list[int], framerate: int) -> None:
        packed = struct.pack(f"<{len(samples)}h", *samples)
        with wave.open(str(file_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(framerate)
            handle.writeframes(packed)
