from ai_layer.common.exceptions import TranscriptionError
from ai_layer.models.transcription_models import AudioClip


class ClipSorter:
    @staticmethod
    def sort_by_start_time(clips: list[AudioClip]) -> list[AudioClip]:
        return sorted(clips, key=lambda clip: clip.time_range.start_time_seconds)

    @staticmethod
    def validate_timestamps(clips: list[AudioClip]) -> None:
        for clip in clips:
            start = clip.time_range.start_time_seconds
            end = clip.time_range.end_time_seconds
            if end < start:
                raise TranscriptionError(
                    f"Clip {clip.clip_id} has invalid time range: {start} -> {end}"
                )
