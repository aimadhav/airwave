"""Deterministic inference adapter for isolated HTTP and queue tests."""

from dataclasses import asdict
from copy import deepcopy
from overlap_lab.exceptions import LabConflict
from overlap_lab.models import ExperimentRequest, LabJob
from overlap_lab.models.listening import ListeningReview


class MockListeningReviewRepository:
    """In-memory labels for deterministic provider-independent tests."""

    def __init__(self, source_hash: str = "a" * 64) -> None:
        self.source_hash = source_hash
        self.data: dict[str, dict[str, object]] = {}

    def reviews(self, recording_id: str) -> dict[str, object]:
        return deepcopy(self.data.get(recording_id, {"recording_id": recording_id,
                        "source_hash": self.source_hash, "revision": 0, "items": {}}))

    def save(self, recording_id: str, source_hash: str, revision: int,
             review: ListeningReview) -> dict[str, object]:
        data = self.reviews(recording_id)
        if source_hash != self.source_hash or revision != data["revision"]:
            raise LabConflict("Stale listening labels")
        data["items"][review.clip.id] = asdict(review)
        data["revision"] += 1
        self.data[recording_id] = data
        return deepcopy(data)


class MockExperimentProcessor:
    def capabilities(self) -> list[dict[str, object]]:
        return [{"id": "sepformer--base", "ready": True, "reason": "Test fixture, not real inference"}]

    def process(self, job_id: str, request: ExperimentRequest) -> dict[str, object]:
        return {"mock": True, "request": asdict(request), "variants": [], "job_id": job_id}

    def load(self, job_id: str) -> LabJob | None:
        return None

    def history(self, recording_id: str) -> list[LabJob]:
        return []
