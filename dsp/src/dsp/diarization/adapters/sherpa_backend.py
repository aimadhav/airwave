"""CPU ONNX adapter: pyannote segmentation + speaker embeddings + clustering."""

import importlib
from typing import Any
import numpy as np

from dsp.diarization.config import LocalDiarizationConfig
from dsp.diarization.interfaces.inference import ModelSpeakerInterval
from dsp.exceptions import DiarizationConfigurationError, DiarizationInferenceError


class SherpaSpeakerBackend:
    def __init__(self, config: LocalDiarizationConfig) -> None:
        config.validate_assets()
        try:
            runtime = importlib.import_module("sherpa_onnx")
        except ImportError as error:
            raise DiarizationConfigurationError(
                "Local diarization requires pip install -r requirements-diarization.txt"
            ) from error
        try:
            runtime_config = runtime.OfflineSpeakerDiarizationConfig(
                segmentation=runtime.OfflineSpeakerSegmentationModelConfig(
                    pyannote=runtime.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=config.segmentation_model, window_shift_ratio=config.window_shift_ratio,
                    ), num_threads=config.num_threads, provider="cpu",
                ),
                embedding=runtime.SpeakerEmbeddingExtractorConfig(
                    model=config.embedding_model, num_threads=config.num_threads, provider="cpu",
                ),
                clustering=runtime.FastClusteringConfig(
                    num_clusters=config.num_speakers, threshold=config.cluster_threshold,
                ),
                min_duration_on=config.min_duration_on, min_duration_off=config.min_duration_off,
            )
            if not runtime_config.validate():
                raise DiarizationConfigurationError("The ONNX diarization configuration is invalid")
            self._engine: Any = runtime.OfflineSpeakerDiarization(runtime_config)
        except DiarizationConfigurationError:
            raise
        except Exception as error:
            raise DiarizationConfigurationError(f"Cannot initialize CPU diarization: {error}") from error

    def infer(self, samples: np.ndarray, sample_rate: int) -> list[ModelSpeakerInterval]:
        if sample_rate != self._engine.sample_rate:
            raise DiarizationInferenceError(f"Model expects {self._engine.sample_rate} Hz, received {sample_rate} Hz")
        result = self._engine.process(samples).sort_by_start_time()
        return [ModelSpeakerInterval(float(s.start), float(s.end), int(s.speaker)) for s in result]
