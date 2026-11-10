"""Optional local speech-recognition adapter; no DSP implementation imports."""

import importlib
from pathlib import Path
import wave
import numpy as np
from recording_viewer.config import RecordingViewerConfig
from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.models import TranscriptWord


class FasterWhisperTranscriber:
    """Decode the mixed recording once with real model-estimated word times."""

    def __init__(self, config: RecordingViewerConfig) -> None:
        self._config = config
        self._model: object | None = None
        if not (Path(config.model_directory) / "model.bin").is_file():
            raise RecordingViewerError("Local recognition weights are missing; inference will not download them")

    def transcribe(self, audio_path: Path) -> list[TranscriptWord]:
        """Return recognized words without guessing missing secondary speech."""
        try:
            # DSP has already decoded this canonical WAV. Read it directly,
            # avoiding a second codec pass and PyAV-version-specific APIs.
            with wave.open(str(audio_path), "rb") as recording:
                if (recording.getframerate(), recording.getnchannels(), recording.getsampwidth()) != (16000, 1, 2):
                    raise RecordingViewerError("Timestamped recognition requires 16 kHz mono PCM16")
                samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").astype(np.float32) / 32768
            return self.transcribe_samples(samples, 16000)
        except RecordingViewerError:
            raise
        except Exception as error:
            raise RecordingViewerError(f"Local timestamped recognition failed: {error}") from error

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptWord]:
        """Reuse one local model for mixed and separated 16 kHz clip inference."""
        from recording_viewer.exceptions import RecoveryError
        try:
            if sample_rate != 16000 or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
                raise RecoveryError("Recognition requires finite 16 kHz mono samples")
            runtime = importlib.import_module("faster_whisper")
            config = self._config
            if self._model is None:
                self._model = runtime.WhisperModel(config.model_directory, device=config.device,
                    compute_type=config.compute_type, cpu_threads=config.cpu_threads, local_files_only=True)
            model = self._model
            segments, _ = model.transcribe(samples, language=config.language,
                                           beam_size=config.beam_size, word_timestamps=True,
                                           vad_filter=config.vad_filter, condition_on_previous_text=config.condition_on_previous_text)
            words: list[TranscriptWord] = []
            for segment in segments:
                if (getattr(segment, "no_speech_prob", 0) > config.maximum_no_speech_probability
                        or getattr(segment, "avg_logprob", 0) < config.minimum_segment_log_probability
                        or getattr(segment, "compression_ratio", 0) > config.maximum_compression_ratio):
                    continue
                for word in segment.words or []:
                    if word.word.strip() and word.end > word.start >= 0:
                        # Clip a model's occasional end-of-file timestamp overshoot.
                        end = min(float(word.end), len(samples) / sample_rate)
                        if end > word.start:
                            words.append(TranscriptWord(float(word.start), end, word.word, getattr(word, "probability", None)))
                print(f"Recognized through {segment.end:.1f}s", flush=True)
            return words
        except Exception as error:
            raise RecoveryError(f"Local timestamped recognition failed: {error}") from error
