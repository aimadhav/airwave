"""Build validated speaker timelines from native or acoustic evidence."""

from dataclasses import replace

from dsp.contracts import MeetTranscriptEntry, SpeakerSegment, VoiceActivitySegment


class TimelineBuilder:
    """Normalize, validate and annotate chronological speaker intervals."""

    def from_meet_entries(
        self,
        entries: list[MeetTranscriptEntry],
        duration_seconds: float | None = None,
    ) -> tuple[list[SpeakerSegment], list[str]]:
        segments = [
            SpeakerSegment(
                segment_id=f"segment_{index:04d}",
                speaker_id=entry.speaker_id,
                start_time=entry.start_time,
                end_time=entry.end_time,
                confidence=1.0 if entry.speaker_id != "SPK_UNKNOWN" else 0.0,
                speaker_source="meet_transcript",
            )
            for index, entry in enumerate(entries)
        ]
        return self._validate_and_mark(segments, duration_seconds)

    def from_vad(
        self,
        segments: list[VoiceActivitySegment],
        speaker_id: str = "SPEAKER_00",
        speaker_source: str = "diarization",
    ) -> tuple[list[SpeakerSegment], list[str]]:
        acoustic_segments = [
            SpeakerSegment(
                segment_id=f"segment_{index:04d}",
                speaker_id=speaker_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker_source=speaker_source,  # type: ignore[arg-type]
            )
            for index, segment in enumerate(segments)
        ]
        return self._validate_and_mark(acoustic_segments, None)

    def validate(
        self,
        segments: list[SpeakerSegment],
        duration_seconds: float | None = None,
    ) -> tuple[list[SpeakerSegment], list[str]]:
        """Validate an already-labelled timeline and mark overlaps."""

        return self._validate_and_mark(segments, duration_seconds)

    def _validate_and_mark(
        self,
        segments: list[SpeakerSegment],
        duration_seconds: float | None,
    ) -> tuple[list[SpeakerSegment], list[str]]:
        warnings: list[str] = []
        valid: list[SpeakerSegment] = []
        for segment in sorted(segments, key=lambda item: (item.start_time, item.end_time)):
            if duration_seconds is not None:
                if segment.start_time >= duration_seconds:
                    warnings.append(f"{segment.segment_id} starts beyond audio duration and was skipped")
                    continue
                if segment.end_time > duration_seconds:
                    warnings.append(f"{segment.segment_id} was clipped to audio duration")
                    segment = replace(segment, end_time=duration_seconds)
            if segment.end_time <= segment.start_time:
                warnings.append(f"{segment.segment_id} has no positive duration and was skipped")
                continue
            valid.append(segment)

        marked: list[SpeakerSegment] = []
        for index, segment in enumerate(valid):
            overlap = any(
                index != other_index
                and segment.start_time < other.end_time
                and other.start_time < segment.end_time
                for other_index, other in enumerate(valid)
            )
            marked.append(replace(segment, is_overlap=overlap))
            if overlap:
                warnings.append(f"{segment.segment_id} overlaps another speaker segment")
        return marked, sorted(set(warnings))
