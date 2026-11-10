import json
import mimetypes
import os
import re
from pathlib import Path

from ai_layer.common.config_loader import STTConfig
from ai_layer.common.exceptions import TranscriptionError
from ai_layer.models.transcription_models import AudioClip


class GeminiSTTProvider:
    """Google Gemini multimodal provider for audio transcription."""

    def __init__(self, config: STTConfig) -> None:
        self._config = config
        self._api_key = (
            os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not self._api_key:
            raise TranscriptionError(
                "Missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini STT provider."
            )

        self._sdk_type = None
        self._client = None
        self._model = None

        # Try modern google-genai SDK first
        try:
            from google import genai

            self._sdk_type = "genai"
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            try:
                import google.generativeai as genai_legacy

                self._sdk_type = "generativeai"
                genai_legacy.configure(api_key=self._api_key)
                self._model = genai_legacy.GenerativeModel(model_name=self._config.model_name)
            except ImportError as exc:
                raise TranscriptionError(
                    "Gemini STT requires google-genai or google-generativeai package. "
                    "Install with: pip install google-genai"
                ) from exc

    def transcribe_clip(self, clip: AudioClip) -> str:
        audio_path = Path(clip.audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio clip not found: {audio_path}")

        prompt = (
            "Generate a precise verbatim transcript of the spoken dialogue in this audio clip. "
            "Return only the spoken text without any commentary, labels, timestamps, or quotation marks."
        )

        try:
            text = self._generate_audio_content(audio_path, prompt)
            return text.strip()
        except Exception as exc:
            raise TranscriptionError(f"Gemini audio transcription failed: {exc}") from exc

    def transcribe_file_segments(self, audio_path: str | Path) -> list[dict]:
        path = Path(audio_path)
        if not path.exists():
            raise TranscriptionError(f"Audio file not found: {path}")

        prompt = (
            "Transcribe the spoken dialogue in this audio recording into timestamped segments. "
            "Return a valid JSON array of objects, where each object has: "
            "'start' (float seconds from start of audio), "
            "'end' (float seconds), and "
            "'text' (verbatim spoken text). "
            "Output JSON only with no markdown formatting or commentary."
        )

        try:
            raw = self._generate_audio_content(path, prompt)
            return self._parse_segments(raw)
        except Exception as exc:
            raise TranscriptionError(f"Gemini timestamped transcription failed: {exc}") from exc

    def _generate_audio_content(self, audio_path: Path, prompt: str) -> str:
        mime_type, _ = mimetypes.guess_type(str(audio_path))
        if not mime_type or not mime_type.startswith("audio/"):
            mime_type = "audio/wav"

        with audio_path.open("rb") as handle:
            audio_bytes = handle.read()

        model_candidates = []
        if getattr(self, "_active_model", None):
            model_candidates.append(self._active_model)
        else:
            if self._config.model_name and "gemini-3" not in self._config.model_name.lower():
                model_candidates.append(self._config.model_name)
            for m in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
                if m not in model_candidates:
                    model_candidates.append(m)

        last_error = None
        for model_name in model_candidates:
            try:
                if self._sdk_type == "genai":
                    from google.genai import types

                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                            prompt,
                        ],
                    )
                    self._active_model = model_name
                    return getattr(response, "text", "") or ""
                elif self._sdk_type == "generativeai":
                    import google.generativeai as genai_legacy

                    target_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
                    model_obj = genai_legacy.GenerativeModel(model_name=target_model)
                    response = model_obj.generate_content([
                        {"mime_type": mime_type, "data": audio_bytes},
                        prompt,
                    ])
                    self._active_model = model_name
                    return getattr(response, "text", "") or ""
            except Exception as exc:
                last_error = exc
                err_msg = str(exc)
                if "404" in err_msg or "not found" in err_msg.lower():
                    continue
                raise

        if last_error:
            raise TranscriptionError(f"Gemini audio processing failed: {last_error}") from last_error

        return ""

    @staticmethod
    def _parse_segments(raw: str) -> list[dict]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        segments: list[dict] = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                segments.append({
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "text": str(item.get("text", "")).strip(),
                })
        return segments
