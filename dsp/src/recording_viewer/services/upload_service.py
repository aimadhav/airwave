"""Bounded local job orchestration depending only on upload contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import PureWindowsPath, PurePosixPath
import threading
from typing import BinaryIO
import uuid
from recording_viewer.exceptions import UploadInputError, UploadQueueFull
from recording_viewer.interfaces.uploads import RecordingProcessor, RecordingUploadStore
from recording_viewer.models.upload import StoredRecording, UploadJob
from recording_viewer.upload_config import UploadConfig


class UploadService:
    """One worker prevents simultaneous recordings exhausting CPU/memory."""

    def __init__(self, store: RecordingUploadStore, processor: RecordingProcessor, config: UploadConfig) -> None:
        self.store, self.processor, self.config = store, processor, config
        self._jobs: dict[str, UploadJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recording-upload")

    def validate(self, filename: str, length: int) -> None:
        """Validate declared inputs before consuming an HTTP request body."""
        if (not filename or any(char in filename for char in '/\\:\x00\r\n')
                or PureWindowsPath(filename).name != filename or PurePosixPath(filename).name != filename
                or PureWindowsPath(filename).is_reserved()
                or filename.endswith((".", " "))):
            raise UploadInputError("Choose a recording with a valid filename")
        if PurePosixPath(filename).suffix.lower() not in self.config.allowed_extensions:
            raise UploadInputError("Unsupported recording format")
        if not 0 < length <= self.config.max_upload_bytes:
            raise UploadInputError("The recording is empty or exceeds the upload size limit")

    def submit(self, filename: str, stream: BinaryIO, length: int) -> UploadJob:
        """Validate before storing, reserve queue capacity and return a job ID."""
        self.validate(filename, length)
        job_id = uuid.uuid4().hex
        job = UploadJob(job_id, filename, "queued", "Waiting for local processing")
        with self._lock:
            if sum(row.status in {"queued", "processing"} for row in self._jobs.values()) >= self.config.max_pending_jobs:
                raise UploadQueueFull("The local queue is full. Please try again after a recording finishes.")
            self._jobs[job_id] = job
        try:
            stored = self.store.save(job_id, filename, stream, length)
            self._executor.submit(self._process, job_id, stored)
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise
        return job

    def get(self, job_id: str) -> UploadJob | None:
        """Read an immutable snapshot while a worker updates the job."""
        with self._lock:
            return self._jobs.get(job_id)

    def _process(self, job_id: str, stored: StoredRecording) -> None:
        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], status="processing",
                                       message="Detecting speakers and transcribing locally. This may take several minutes.")
        try:
            self.processor.process(job_id, stored)
            with self._lock:
                self._jobs[job_id] = replace(self._jobs[job_id], status="completed", message="Your transcript is ready",
                                           result_url=f"/recordings/{job_id}/index.html")
        except Exception:
            with self._lock:
                self._jobs[job_id] = replace(self._jobs[job_id], status="failed",
                    message="Could not process this recording. Check the file format, duration limit and local model setup. Private details are in the processing log.")

    def close(self) -> None:
        """Drain accepted work cleanly when the local server shuts down."""
        self._executor.shutdown(wait=True)
