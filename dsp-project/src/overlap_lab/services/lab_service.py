"""Bounded asynchronous comparison jobs, depending only on abstract providers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import uuid
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabBusy, LabError, LabInputError
from overlap_lab.interfaces import ExperimentProcessor, StudyRepository
from overlap_lab.models import ExperimentRequest, LabJob, ReferenceAnnotation
from overlap_lab.services.scoring import tokenize


class LabService:
    def __init__(self, repository: StudyRepository, processor: ExperimentProcessor, config: LabConfig) -> None:
        self.repository, self.processor, self.config = repository, processor, config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="overlap-lab")
        self._lock = threading.Lock()
        self._jobs: dict[str, LabJob] = {}

    def validate_reference(self, recording_id: str, reference: ReferenceAnnotation) -> None:
        study = self.repository.study(recording_id)
        if reference.clip.end > study["duration"] or reference.clip.end - reference.clip.start > self.config.max_clip_seconds:
            raise LabInputError("Choose a clip within the recording and duration limit")
        if not set(reference.clip.speakers).issubset(set(study["speakers"])):
            raise LabInputError("Speaker IDs must belong to this recording")
        active = {r["speaker"] for r in study["regions"]
                  if min(r["end"], reference.clip.end) > max(r["start"], reference.clip.start)}
        if active != set(reference.clip.speakers):
            raise LabInputError("The interval must contain exactly the two selected speakers. Adjust its bounds.")
        if any(len(text) > self.config.max_reference_characters for text in reference.texts.values()):
            raise LabInputError("Reference transcript is too long")
        if any(len(tokenize(text)) > self.config.max_reference_words for text in reference.texts.values()):
            raise LabInputError("Reference has too many words for a short clip")

    def submit(self, request: ExperimentRequest) -> LabJob:
        """Only installed, registered combinations may enter the queue."""
        self.validate_reference(request.recording_id, request.reference)
        if request.context_seconds > self.config.max_context_seconds:
            raise LabInputError("Context exceeds the configured limit")
        available = {row["id"] for row in self.processor.capabilities() if row["ready"]}
        if not set(request.variants).issubset(available):
            raise LabInputError("A selected model is unavailable; install/connect it first")
        job = LabJob(uuid.uuid4().hex, request.recording_id, "queued", "Waiting for local CPU")
        with self._lock:
            if sum(j.status in {"queued", "running"} for j in self._jobs.values()) >= self.config.max_pending_jobs:
                raise LabBusy("The experiment queue is full. Let the current comparison finish.")
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id, request)
        return job

    def get(self, job_id: str) -> LabJob | None:
        with self._lock:
            current = self._jobs.get(job_id)
        return current or self.processor.load(job_id)

    def history(self, recording_id: str) -> list[LabJob]:
        """Show active jobs first, then persisted recency, without stale shadows."""
        stored = {job.id: job for job in self.processor.history(recording_id)}
        with self._lock:
            remembered = [job for job in reversed(list(self._jobs.values())) if job.recording_id == recording_id]
        active = [job for job in remembered if job.status in {"queued", "running"}]
        # Completed jobs cached by this server may be older than newly persisted
        # CLI experiments. Do not let them push recent comparisons out of view.
        ephemeral = [job for job in remembered if job.status not in {"queued", "running"}
                     and job.id not in stored and self.processor.load(job.id) is None]
        active_ids = {job.id for job in active}
        return [*active, *ephemeral, *(job for job in stored.values() if job.id not in active_ids)][:self.config.max_history]

    def _run(self, job_id: str, request: ExperimentRequest) -> None:
        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], status="running", message="Separating and transcribing locally…")
        try:
            result = self.processor.process(job_id, request)
            with self._lock:
                self._jobs[job_id] = replace(self._jobs[job_id], status="completed", message="Comparison ready", result=result)
        except Exception:
            with self._lock:
                self._jobs[job_id] = replace(self._jobs[job_id], status="failed",
                    message="Experiment failed. References are safe; private details are in its local log.")

    def close(self) -> None:
        self._executor.shutdown(wait=True)
