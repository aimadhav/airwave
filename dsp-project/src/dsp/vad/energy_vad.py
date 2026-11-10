"""Deterministic energy-based voice activity detector."""

from dataclasses import dataclass

import numpy as np

from dsp.contracts import AudioBuffer, VoiceActivitySegment


@dataclass(frozen=True)
class EnergyVAD:
    """Detect speech using short-time RMS energy.

    This is the dependency-free baseline. A model-based provider can later
    implement the same output contract without changing downstream stages.
    """

    frame_length_ms: float = 30.0
    hop_length_ms: float = 10.0
    threshold_db: float = -35.0
    min_speech_ms: float = 200.0
    min_silence_ms: float = 250.0
    adaptive_noise_floor: bool = True
    noise_floor_percentile: float = 10.0
    noise_margin_db: float = 3.0

    def detect(self, audio: AudioBuffer) -> list[VoiceActivitySegment]:
        """Return speech intervals for mono audio."""

        if audio.metadata.channels != 1:
            raise ValueError("EnergyVAD requires mono audio")
        sample_rate = audio.metadata.sample_rate
        frame_length = max(1, round(sample_rate * self.frame_length_ms / 1000))
        hop_length = max(1, round(sample_rate * self.hop_length_ms / 1000))
        if hop_length > frame_length:
            raise ValueError("hop_length_ms cannot exceed frame_length_ms")

        signal = audio.samples[:, 0]
        if signal.size == 0:
            return []
        starts = list(range(0, max(1, signal.size - frame_length + 1), hop_length))
        if not starts or starts[-1] + frame_length < signal.size:
            starts.append(max(0, signal.size - frame_length))
        energies = np.asarray(
            [np.sqrt(np.mean(np.square(signal[start : start + frame_length]))) for start in starts]
        )
        db = 20.0 * np.log10(np.maximum(energies, 1e-8))
        threshold_db = self._effective_threshold(db)
        active = db >= threshold_db

        intervals = self._active_intervals(active, starts, frame_length, hop_length, sample_rate)
        frame_times = np.asarray(starts, dtype=np.float64) / sample_rate
        return [
            VoiceActivitySegment(
                start_time=start,
                end_time=end,
                confidence=self._confidence(db, frame_times, start, end),
            )
            for start, end in intervals
        ]

    def _effective_threshold(self, db: np.ndarray) -> float:
        """Choose a stable threshold while adapting to a noisy recording.

        The fixed threshold remains the lower bound so quiet recordings do not
        become over-sensitive. For noisy or compressed recordings, a low
        percentile of the measured frame energies acts as a noise-floor
        estimate and raises the threshold by a small configurable margin.
        """

        if not self.adaptive_noise_floor or db.size == 0:
            return self.threshold_db
        percentile = float(np.clip(self.noise_floor_percentile, 0.0, 100.0))
        noise_floor_db = float(np.percentile(db, percentile))
        return max(self.threshold_db, noise_floor_db + self.noise_margin_db)

    def _active_intervals(
        self,
        active: np.ndarray,
        starts: list[int],
        frame_length: int,
        hop_length: int,
        sample_rate: int,
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        start_index: int | None = None
        max_gap_frames = max(0, round(self.min_silence_ms / self.hop_length_ms))
        gap_frames = 0

        for index, is_active in enumerate(active):
            if is_active:
                if start_index is None:
                    start_index = index
                gap_frames = 0
                continue
            if start_index is not None:
                gap_frames += 1
                if gap_frames > max_gap_frames:
                    intervals.append(self._interval(start_index, index - gap_frames + 1, starts, frame_length, sample_rate))
                    start_index = None
                    gap_frames = 0

        if start_index is not None:
            intervals.append(self._interval(start_index, len(active) - 1, starts, frame_length, sample_rate))

        minimum_duration = self.min_speech_ms / 1000.0
        return [interval for interval in intervals if interval[1] - interval[0] >= minimum_duration]

    @staticmethod
    def _interval(
        start_index: int,
        end_index: int,
        starts: list[int],
        frame_length: int,
        sample_rate: int,
    ) -> tuple[float, float]:
        start = starts[start_index] / sample_rate
        end = (starts[end_index] + frame_length) / sample_rate
        return start, end

    @staticmethod
    def _confidence(db: np.ndarray, frame_times: np.ndarray, start: float, end: float) -> float:
        """Map mean energy above threshold to a bounded diagnostic score."""

        active_db = db[(frame_times >= start) & (frame_times <= end) & np.isfinite(db)]
        if active_db.size == 0:
            return 0.0
        # This is a relative energy confidence, not a speaker confidence.
        score = (float(np.mean(active_db)) + 80.0) / 80.0
        return float(np.clip(score, 0.0, 1.0))
