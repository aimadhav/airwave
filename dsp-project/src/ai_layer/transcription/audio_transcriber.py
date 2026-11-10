from ai_layer.common.stt_provider import STTProvider
from ai_layer.models.transcription_models import AudioClip, RawUtterance


class AudioTranscriber:
    def __init__(self, stt_provider: STTProvider) -> None:
        self._stt_provider = stt_provider

    def transcribe_single_clip(self, clip: AudioClip) -> RawUtterance:
        text = self._stt_provider.transcribe_clip(clip)
        return RawUtterance(
            utterance_id=f"utt_{clip.clip_id}",
            speaker_id=clip.preliminary_speaker_id,
            text=text,
            time_range=clip.time_range,
            audio_path=clip.audio_path,
        )

    def transcribe_batch(self, clips: list[AudioClip]) -> list[RawUtterance]:
        return [self.transcribe_single_clip(clip) for clip in clips]
