"""Model-driven diarization independent of the chosen runtime library."""

import math
import numpy as np

from dsp.contracts import AudioBuffer, SpeakerSegment, VoiceActivitySegment
from dsp.diarization.interfaces.inference import SpeakerInferenceBackend
from dsp.exceptions import DiarizationInferenceError


class LocalModelDiarizer:
    """Use neural segmentation rather than gating speech with energy VAD.

    Energy-VAD regions remain diagnostic hints. They are not a hard gate: a
    loud background can defeat an energy detector while the model detects voice.
    Cluster IDs are renamed in first-occurrence order; real names are never inferred.
    The backend has no calibrated confidence, so 0 denotes unavailable confidence.
    """

    def __init__(self, backend: SpeakerInferenceBackend) -> None:
        self._backend = backend

    def diarize(self, audio: AudioBuffer, speech_segments: list[VoiceActivitySegment]) -> list[SpeakerSegment]:
        if audio.metadata.channels != 1 or audio.metadata.sample_rate != 16_000:
            raise DiarizationInferenceError("Local diarization requires canonical 16 kHz mono audio")
        samples = np.ascontiguousarray(audio.samples[:, 0], dtype=np.float32)
        if not np.all(np.isfinite(samples)):
            raise DiarizationInferenceError("Audio contains non-finite samples")
        if samples.size == 0 or not np.any(samples):
            return []
        try:
            intervals = sorted(self._backend.infer(samples, audio.metadata.sample_rate), key=lambda s: (s.start, s.end, s.cluster))
        except Exception as error:
            raise DiarizationInferenceError(f"Local speaker inference failed: {error}") from error
        names: dict[int, str] = {}
        result: list[SpeakerSegment] = []
        for row in intervals:
            if not math.isfinite(row.start) or not math.isfinite(row.end) or row.start < 0 or row.end <= row.start or row.cluster < 0:
                raise DiarizationInferenceError("Model returned an invalid speaker interval")
            start, end = row.start, min(row.end, audio.duration_seconds)
            if end <= start:
                continue
            if row.cluster not in names:
                names[row.cluster] = f"SPEAKER_{len(names):02d}"
            result.append(SpeakerSegment(
                segment_id=f"segment_{len(result):04d}", speaker_id=names[row.cluster],
                start_time=start, end_time=end, confidence=0.0, speaker_source="diarization",
            ))
        return result
