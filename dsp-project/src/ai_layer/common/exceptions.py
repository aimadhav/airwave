class AILayerError(Exception):
    """Base exception for AI Layer failures."""


class TranscriptionError(AILayerError):
    """Raised when transcription fails."""


class SpeakerVerificationError(AILayerError):
    """Raised when speaker verification fails."""


class AnalysisExtractionError(AILayerError):
    """Raised when insight extraction fails."""


class KnowledgeMemoryError(AILayerError):
    """Raised when knowledge integration fails."""


class SummarizationError(AILayerError):
    """Raised when summary generation fails."""


class QueryAnsweringError(AILayerError):
    """Raised when query answering fails."""
