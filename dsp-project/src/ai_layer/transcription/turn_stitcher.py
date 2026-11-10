from ai_layer.models.transcription_models import RawUtterance, StitchedTurn


class TurnStitcher:
    def __init__(self, max_pause_seconds: float) -> None:
        self._max_pause_seconds = max_pause_seconds

    def stitch_consecutive_turns(self, utterances: list[RawUtterance]) -> list[StitchedTurn]:
        if not utterances:
            return []

        turns: list[StitchedTurn] = []
        current = utterances[0]
        source_ids = [current.utterance_id]
        turn_index = 1

        for utterance in utterances[1:]:
            pause = utterance.time_range.start_time_seconds - current.time_range.end_time_seconds
            same_speaker = utterance.speaker_id == current.speaker_id
            if same_speaker and pause <= self._max_pause_seconds:
                current = RawUtterance(
                    utterance_id=current.utterance_id,
                    speaker_id=current.speaker_id,
                    text=f"{current.text} {utterance.text}".strip(),
                    time_range=current.time_range.__class__(
                        start_time_seconds=current.time_range.start_time_seconds,
                        end_time_seconds=utterance.time_range.end_time_seconds,
                    ),
                    audio_path=current.audio_path,
                )
                source_ids.append(utterance.utterance_id)
                continue

            turns.append(
                StitchedTurn(
                    turn_id=f"turn_{turn_index}",
                    speaker_id=current.speaker_id,
                    text=current.text,
                    time_range=current.time_range,
                    source_utterance_ids=list(source_ids),
                )
            )
            turn_index += 1
            current = utterance
            source_ids = [utterance.utterance_id]

        turns.append(
            StitchedTurn(
                turn_id=f"turn_{turn_index}",
                speaker_id=current.speaker_id,
                text=current.text,
                time_range=current.time_range,
                source_utterance_ids=list(source_ids),
            )
        )
        return turns
