"""Domain errors for the retrieval layer."""


class UpscAssistantError(Exception):
    """Base error for the application."""


class DatasetError(UpscAssistantError):
    """Raised when source data is malformed or inconsistent."""


class RetrievalError(UpscAssistantError):
    """Raised when a retrieval request is invalid."""
