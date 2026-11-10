"""DSP-specific exception types."""


class DSPError(Exception):
    """Base class for expected DSP failures."""


class AudioLoadError(DSPError):
    """Raised when an input audio file cannot be decoded."""


class UnsupportedAudioError(DSPError):
    """Raised when the input format is not supported by the current stage."""


class DiarizationConfigurationError(DSPError):
    """A diarization provider, dependency or model asset is unavailable."""


class DiarizationInferenceError(DSPError):
    """A configured diarization model failed to produce a valid timeline."""
