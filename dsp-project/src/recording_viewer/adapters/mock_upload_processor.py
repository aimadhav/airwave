"""Explicit fixture-only upload processor for isolated integration tests."""

from recording_viewer.models.upload import StoredRecording


class MockRecordingProcessor:
    """Produce a clearly labelled mock result without models or APIs."""

    def process(self, job_id: str, recording: StoredRecording) -> None:
        """Write a test marker, never pretend fixture output is transcription."""
        recording.result.mkdir(parents=True, exist_ok=False)
        (recording.result / "index.html").write_text("<!doctype html><title>Mock upload test</title><p>Fixture only; no recognition performed.</p>", encoding="utf-8")
