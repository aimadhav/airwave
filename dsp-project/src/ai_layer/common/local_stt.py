"""Optional offline English Whisper STT adapter; real audio, never scripted text."""

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any
import wave
import math
import numpy as np

from ai_layer.models.transcription_models import AudioClip
from ai_layer.models.base_models import TimeRange


class LocalSTTError(RuntimeError):
    """A missing local speech model, invalid clip, or decoding failure."""


@dataclass(frozen=True)
class LocalSTTConfig:
    encoder: str = "models/stt/tiny.en-encoder.int8.onnx"
    decoder: str = "models/stt/tiny.en-decoder.int8.onnx"
    tokens: str = "models/stt/tiny.en-tokens.txt"
    num_threads: int = 2
    max_chunk_seconds: float = 25.0

    def __post_init__(self) -> None:
        if self.num_threads < 1:
            raise LocalSTTError("STT threads must be positive")
        if not math.isfinite(self.max_chunk_seconds) or not 0 < self.max_chunk_seconds <= 30:
            raise LocalSTTError("Whisper chunk duration must be positive and at most 30 seconds")

    def validate_assets(self) -> None:
        for asset in (self.encoder, self.decoder, self.tokens):
            if not Path(asset).is_file():
                raise LocalSTTError(f"Missing local Whisper asset: {asset}. Run python scripts/setup_diarization.py --with-stt")


class LocalWhisperSTTProvider:
    """Implements STTProvider using only CPU and downloaded model files.

    Clips are already normalized by DSP. This boundary reads WAV directly,
    avoiding a dependency on DSP internals. No calibrated word confidence is
    supplied. Overlap is not unmixed; transcription may omit one voice.
    """

    def __init__(self, config: LocalSTTConfig) -> None:
        config.validate_assets()
        self._config = config
        try:
            runtime = importlib.import_module("sherpa_onnx")
            self._engine: Any = runtime.OfflineRecognizer.from_whisper(
                encoder=config.encoder, decoder=config.decoder, tokens=config.tokens,
                num_threads=config.num_threads, provider="cpu", debug=False,
            )
        except Exception as error:
            raise LocalSTTError(f"Cannot initialize offline Whisper: {error}") from error

    def transcribe_clip(self, clip: AudioClip) -> str:
        try:
            with wave.open(clip.audio_path, "rb") as source:
                if source.getsampwidth() != 2 or source.getnchannels() != 1:
                    raise LocalSTTError("Offline STT expects mono 16-bit PCM WAV clips from DSP")
                rate = source.getframerate()
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
            if not samples.size or not np.any(samples):
                return ""
            # Whisper has a finite acoustic context. Decode all of a long clip,
            # rather than silently discarding everything after that context.
            chunk_size = max(1, int(rate * self._config.max_chunk_seconds))
            parts: list[str] = []
            for start in range(0, len(samples), chunk_size):
                stream = self._engine.create_stream()
                stream.accept_waveform(rate, samples[start:start + chunk_size])
                self._engine.decode_stream(stream)
                text = str(stream.result.text).strip()
                if text:
                    parts.append(text)
            return " ".join(parts)
        except LocalSTTError:
            raise
        except Exception as error:
            raise LocalSTTError(f"Failed to transcribe {clip.clip_id}: {error}") from error

    def transcribe_file_segments(self, audio_path: str) -> list[dict[str, object]]:
        """Honor whole-file STT without pretending it includes diarization."""
        try:
            with wave.open(audio_path, "rb") as source:
                duration = source.getnframes() / source.getframerate()
            if duration <= 0:
                return []
            text = self.transcribe_clip(AudioClip("whole_file", audio_path, "SPK_UNKNOWN", TimeRange(0, duration)))
            return [{"start": 0.0, "end": duration, "text": text, "speaker_id": "SPK_UNKNOWN"}] if text else []
        except LocalSTTError:
            raise
        except Exception as error:
            raise LocalSTTError(f"Cannot load local STT input: {error}") from error
