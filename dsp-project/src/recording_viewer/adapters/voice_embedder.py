"""Optional CPU ONNX embeddings, independent of the diarization implementation."""

import importlib
from math import gcd
from pathlib import Path
import numpy as np
from recording_viewer.exceptions import RecoveryError
from recording_viewer.recovery_config import RecoveryConfig


class OnnxVoiceEmbedder:
    """No inference-time downloads or registered real-person identities."""

    def __init__(self, config: RecoveryConfig) -> None:
        if not Path(config.embedding_model).is_file():
            raise RecoveryError("Local voice embedding model is missing")
        try:
            self.bandlimit_rate = config.separator_sample_rate
            self.resample = importlib.import_module("scipy.signal").resample_poly
            runtime = importlib.import_module("sherpa_onnx")
            self.engine = runtime.SpeakerEmbeddingExtractor(runtime.SpeakerEmbeddingExtractorConfig(
                model=config.embedding_model, num_threads=config.embedding_threads, provider="cpu"))
        except Exception as error:
            raise RecoveryError(f"Cannot initialize voice embeddings: {error}") from error

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute a vector; silence/model failures surface as typed errors."""
        try:
            # Separation bandwidth differs from clean wideband audio. Compare
            # like with like: antialiased down/up sampling of BOTH references
            # and separated tracks avoids a systematic identity-domain shift.
            values = np.asarray(samples, dtype=np.float32)
            if self.bandlimit_rate < sample_rate:
                factor = gcd(self.bandlimit_rate, sample_rate)
                values = self.resample(self.resample(values, self.bandlimit_rate // factor, sample_rate // factor),
                                       sample_rate // factor, self.bandlimit_rate // factor).astype(np.float32)
            stream = self.engine.create_stream()
            stream.accept_waveform(sample_rate, values)
            stream.input_finished()
            if not self.engine.is_ready(stream):
                raise RecoveryError("Insufficient speech for a voice embedding")
            return np.asarray(self.engine.compute(stream), dtype=np.float64)
        except RecoveryError:
            raise
        except Exception as error:
            raise RecoveryError(f"Voice embedding inference failed: {error}") from error
