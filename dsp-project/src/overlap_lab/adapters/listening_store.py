"""Source-bound atomic storage for lightweight listening reviews only."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import threading
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabConflict, LabInputError
from overlap_lab.interfaces import StudyRepository
from overlap_lab.models.listening import ListeningReview


class FileListeningReviewRepository:
    """Never reads or rewrites the transcript-reference annotation files."""

    def __init__(self, studies: StudyRepository, storage: Path, config: LabConfig) -> None:
        self.studies, self.storage, self.config = studies, storage.resolve(), config
        self._lock = threading.Lock()
        self.storage.mkdir(parents=True, exist_ok=True)

    def _identity(self, recording_id: str) -> tuple[Path, dict[str, object]]:
        if not isinstance(recording_id, str) or not re.fullmatch(r"current|[a-f0-9]{32}", recording_id):
            raise LabInputError("Unknown recording identifier")
        study = self.studies.study(recording_id)
        source_hash = study["source_hash"]
        if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
            raise LabInputError("Invalid recording source identity")
        empty = {"recording_id": recording_id, "source_hash": source_hash, "revision": 0, "items": {}}
        return self.storage / f"{recording_id}-{source_hash}.json", empty

    def _read(self, path: Path, empty: dict[str, object]) -> dict[str, object]:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else empty
            if (data["source_hash"] != empty["source_hash"] or data["recording_id"] != empty["recording_id"]
                    or type(data["revision"]) is not int or data["revision"] < 0 or not isinstance(data["items"], dict)):
                raise ValueError("Invalid saved review state")
            return data
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise LabInputError("Saved listening labels could not be read; no files were changed") from error

    def reviews(self, recording_id: str) -> dict[str, object]:
        """Load this source's listening labels without requiring a speech model."""
        path, empty = self._identity(recording_id)
        with self._lock:
            return self._read(path, empty)

    def save(self, recording_id: str, source_hash: str, revision: int,
             review: ListeningReview) -> dict[str, object]:
        """Compare-and-swap protects edits made in another browser tab."""
        path, empty = self._identity(recording_id)
        if source_hash != empty["source_hash"]:
            raise LabConflict("Recording source changed. Reload before saving a listening label.")
        with self._lock:
            data = self._read(path, empty)
            if data["revision"] != revision:
                raise LabConflict("Another tab saved newer listening labels. Reload saved labels, then try again.")
            if review.clip.id not in data["items"] and len(data["items"]) >= self.config.max_annotations:
                raise LabInputError("Listening-label limit reached for this recording")
            data["items"][review.clip.id] = asdict(review)
            data["revision"] += 1
            temporary = path.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(path)
            except OSError as error:
                raise LabInputError("Listening label could not be saved; previous saved labels are unchanged") from error
            return data
