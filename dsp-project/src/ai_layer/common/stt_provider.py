from typing import Protocol

from ai_layer.models.transcription_models import AudioClip


class STTProvider(Protocol):
    def transcribe_clip(self, clip: AudioClip) -> str:
        ...

    def transcribe_file_segments(self, audio_path: str) -> list[dict]:
        ...


class MockSTTProvider:
    """Deterministic STT mock keyed by clip metadata."""

    _DEFAULT_TRANSCRIPTS: dict[str, str] = {
        "clip_1": "Hello everyone, thanks for joining.",
        "clip_2": "Hi, I'm Sarah from the product team.",
        "clip_3": "The client renewed the annual contract on Monday.",
        "clip_4": "We decided to reschedule the launch date to October 15.",
        "clip_5": "David will prepare the slide deck by Friday.",
    }

    def transcribe_clip(self, clip: AudioClip) -> str:
        if clip.clip_id in self._DEFAULT_TRANSCRIPTS:
            return self._DEFAULT_TRANSCRIPTS[clip.clip_id]

        speaker = clip.preliminary_speaker_id.replace("_", " ")
        return f"Transcribed speech from {speaker}."

    def transcribe_file_segments(self, audio_path: str) -> list[dict]:
        return [
            {"start": 0.0, "end": 3.0, "text": "Hello everyone, thanks for joining.", "speaker_id": "Speaker_1"},
            {"start": 3.5, "end": 7.0, "text": "Hi, I'm Sarah from the product team.", "speaker_id": "Speaker_2"},
            {"start": 7.5, "end": 11.0, "text": "The client renewed the annual contract on Monday.", "speaker_id": "Speaker_2"},
            {"start": 11.5, "end": 15.0, "text": "We decided to reschedule the launch date to October 15.", "speaker_id": "Speaker_1"},
            {"start": 15.5, "end": 19.0, "text": "David will prepare the slide deck by Friday.", "speaker_id": "Speaker_3"},
        ]


class MultiSpeakerDemoSTTProvider:
    """Offline scripted STT provider used only by the multi-speaker demo.

    The demo audio is deterministic acoustic synthesis, not human speech. A
    scripted provider lets the web demonstration show the complete speaker
    timeline and AI handoff without pretending that tones contain real words.
    """

    _SCRIPT: tuple[str, ...] = (
        "Welcome everyone. Let's review the release plan.",
        "The backend migration is complete and the tests are passing.",
        "I will finish the security review by Friday.",
        "Great. We agreed to target the October 15 release.",
    )

    def transcribe_clip(self, clip: AudioClip) -> str:
        digits = "".join(character for character in clip.clip_id if character.isdigit())
        index = int(digits[-4:]) if digits else 0
        return self._SCRIPT[index % len(self._SCRIPT)]

    def transcribe_file_segments(self, audio_path: str) -> list[dict]:
        return [
            {"start": index * 4.0, "end": (index + 1) * 4.0, "text": text, "speaker_id": f"SPEAKER_{index % 3:02d}"}
            for index, text in enumerate(self._SCRIPT)
        ]
