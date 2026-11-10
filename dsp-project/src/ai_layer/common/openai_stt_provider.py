import os
from pathlib import Path

from ai_layer.common.config_loader import STTConfig
from ai_layer.common.exceptions import TranscriptionError
from ai_layer.models.transcription_models import AudioClip


class OpenAISTTProvider:
    """OpenAI Whisper API provider for per-clip transcription."""

    def __init__(self, config: STTConfig) -> None:
        self._config = config
        api_key_env = "OPENAI_API_KEY"
        self._api_key = os.environ.get(api_key_env, "")
        if not self._api_key:
            raise TranscriptionError(f"Missing API key: set {api_key_env} for OpenAI STT provider.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise TranscriptionError(
                "OpenAI STT requires the openai package. Install with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self._api_key)

    def transcribe_clip(self, clip: AudioClip) -> str:
        audio_path = Path(clip.audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio clip not found: {audio_path}")

        with audio_path.open("rb") as handle:
            response = self._client.audio.transcriptions.create(
                model=self._config.model_name,
                file=handle,
                language=self._config.language,
            )
        text = getattr(response, "text", "") or ""
        return text.strip()

    def transcribe_file_segments(self, audio_path: str | Path) -> list[dict]:
        path = Path(audio_path)
        if not path.exists():
            raise TranscriptionError(f"Audio file not found: {path}")

        with path.open("rb") as handle:
            response = self._client.audio.transcriptions.create(
                model=self._config.model_name,
                file=handle,
                language=self._config.language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments: list[dict] = []
        raw_segments = getattr(response, "segments", None) or []
        for s in raw_segments:
            start = float(getattr(s, "start", 0.0) if hasattr(s, "start") else s.get("start", 0.0))
            end = float(getattr(s, "end", 0.0) if hasattr(s, "end") else s.get("end", 0.0))
            text = str(getattr(s, "text", "") if hasattr(s, "text") else s.get("text", "")).strip()
            if text:
                segments.append({"start": start, "end": end, "text": text})
        return segments

