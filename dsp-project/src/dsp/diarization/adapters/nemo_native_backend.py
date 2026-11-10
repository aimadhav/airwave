"""Optional native CPU diarizer behind the shared speaker-inference protocol."""

import json
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np

from dsp.diarization.interfaces.inference import ModelSpeakerInterval
from dsp.diarization.native_config import NativeDiarizationConfig
from dsp.exceptions import DiarizationInferenceError


def parse_native_segments(text: str) -> list[ModelSpeakerInterval]:
    """Validate CLI JSON at the boundary; preserve simultaneous speaker activity."""
    try:
        document = json.loads(text)
        if not isinstance(document, dict) or not isinstance(document.get("segments"), list):
            raise ValueError("Expected an object containing a segments list")
        intervals: list[ModelSpeakerInterval] = []
        for row in document["segments"]:
            if not isinstance(row, dict):
                raise ValueError("Invalid segment object")
            start, end, speaker = row["start"], row["end"], row["speaker"]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
                raise ValueError("Invalid segment timestamps")
            if start < 0 or end <= start or not isinstance(speaker, int) or isinstance(speaker, bool) or not 1 <= speaker <= 8:
                raise ValueError("Invalid interval or V3 speaker index")
            intervals.append(ModelSpeakerInterval(float(start), float(end), speaker - 1))
        return intervals
    except (ValueError, TypeError, KeyError) as error:
        raise DiarizationInferenceError(f"Invalid native diarization response: {error}") from error


class NemoNativeSpeakerBackend:
    """Use local Nemotron weights; never load ASR or call a remote service."""

    def __init__(self, config: NativeDiarizationConfig) -> None:
        config.validate_assets()
        self._config = config

    def infer(self, samples: np.ndarray, sample_rate: int) -> list[ModelSpeakerInterval]:
        """Serialize canonical audio temporarily and bound subprocess execution."""
        if sample_rate != 16000 or samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
            raise DiarizationInferenceError("Native diarization requires finite 16 kHz mono samples")
        config = self._config
        if config.full_attention and len(samples) / sample_rate > config.max_full_attention_seconds:
            raise DiarizationInferenceError("Full-attention duration limit exceeded; use v3-offline chunked inference")
        try:
            with tempfile.TemporaryDirectory(prefix="dsp-native-") as temporary:
                input_path = Path(temporary) / "input.wav"
                pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
                with wave.open(str(input_path), "wb") as recording:
                    recording.setparams((1, 2, sample_rate, len(pcm), "NONE", "not compressed"))
                    recording.writeframes(pcm.tobytes())
                command = [str(Path(config.executable).resolve()), "diarize", str(input_path),
                           "--model", str(Path(config.model).resolve()), "--device", "cpu",
                           "--preset", config.preset, "--format", "json"]
                if config.full_attention:
                    command.append("--offline")
                for field, flag in (("onset", "--onset"), ("offset", "--offset"),
                                    ("pad_onset_seconds", "--pad-onset"), ("pad_offset_seconds", "--pad-offset"),
                                    ("min_duration_on", "--min-duration-on"), ("min_duration_off", "--min-duration-off")):
                    value = getattr(config, field)
                    if value is not None:
                        command.extend([flag, str(value)])
                completed = subprocess.run(command, check=True, capture_output=True, text=True,
                                           encoding="utf-8", timeout=config.timeout_seconds, shell=False)
                return parse_native_segments(completed.stdout)
        except subprocess.TimeoutExpired as error:
            raise DiarizationInferenceError(f"Native diarization exceeded {config.timeout_seconds}s timeout") from error
        except subprocess.CalledProcessError as error:
            raise DiarizationInferenceError(f"Native diarization failed: {(error.stderr or '')[-2000:]}") from error
        except OSError as error:
            raise DiarizationInferenceError(f"Cannot run native diarization: {error}") from error
