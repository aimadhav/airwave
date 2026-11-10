"""Errors exposed by the recording viewer boundary."""


class RecordingViewerError(Exception):
    """Base error for invalid viewer inputs or local processing failures."""


class TranscriptInputError(RecordingViewerError):
    """A word or speaker interval violates the timestamp contract."""


class RecoveryError(RecordingViewerError):
    """Overlap inference failed or violated the recovery contract."""


class UploadInputError(RecordingViewerError):
    """A recording upload is invalid or incomplete."""


class UploadQueueFull(RecordingViewerError):
    """The bounded local processing queue is currently full."""


class VoiceProfileError(RecordingViewerError):
    """Persistent voice-profile data or local storage violates its contract."""
