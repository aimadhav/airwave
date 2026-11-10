"""Diarization providers used by the offline-first DSP pipeline."""

from typing import Protocol

from dsp.contracts import AudioBuffer, SpeakerSegment, VoiceActivitySegment
from dsp.timeline import TimelineBuilder


class DiarizationProvider(Protocol):
    """Provider contract for assigning anonymous speaker IDs to speech."""

    def diarize(
        self,
        audio: AudioBuffer,
        speech_segments: list[VoiceActivitySegment],
    ) -> list[SpeakerSegment]:
        """Return timestamped speaker segments."""


class SingleSpeakerDiarizer:
    """Safe fallback that labels all detected speech as one anonymous speaker."""

    def diarize(
        self,
        audio: AudioBuffer,
        speech_segments: list[VoiceActivitySegment],
    ) -> list[SpeakerSegment]:
        segments, _ = TimelineBuilder().from_vad(
            speech_segments,
            speaker_id="SPEAKER_00",
            speaker_source="unknown",
        )
        return segments


class MockDiarizer:
    """Deterministic multi-speaker provider for tests and demonstrations."""

    def __init__(self, speaker_count: int = 2) -> None:
        if speaker_count < 2:
            raise ValueError("speaker_count must be at least 2")
        self.speaker_count = speaker_count

    def diarize(
        self,
        audio: AudioBuffer,
        speech_segments: list[VoiceActivitySegment],
    ) -> list[SpeakerSegment]:
        result: list[SpeakerSegment] = []
        for index, segment in enumerate(speech_segments):
            result.append(
                SpeakerSegment(
                    segment_id=f"segment_{index:04d}",
                    speaker_id=f"SPEAKER_{index % self.speaker_count:02d}",
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    speaker_source="diarization",
                )
            )
        return result
