from ai_layer.common.config_loader import TranscriptionConfig
from ai_layer.common.stt_provider import STTProvider
from ai_layer.models.transcription_models import AudioClip, STTTranscriptOutput
from ai_layer.transcription.audio_transcriber import AudioTranscriber
from ai_layer.transcription.clip_sorter import ClipSorter
from ai_layer.transcription.turn_stitcher import TurnStitcher


class STTTranscriptionService:
    def __init__(self, stt_provider: STTProvider, config: TranscriptionConfig) -> None:
        self._transcriber = AudioTranscriber(stt_provider)
        self._turn_stitcher = TurnStitcher(config.max_pause_seconds)

    def process_session_clips(
        self,
        session_id: str,
        clips: list[AudioClip],
    ) -> STTTranscriptOutput:
        ClipSorter.validate_timestamps(clips)
        sorted_clips = ClipSorter.sort_by_start_time(clips)
        raw_utterances = self._transcriber.transcribe_batch(sorted_clips)
        turns = self._turn_stitcher.stitch_consecutive_turns(raw_utterances)
        return STTTranscriptOutput(
            session_id=session_id,
            turns=turns,
            raw_utterances=raw_utterances,
        )
