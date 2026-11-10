"""Allowlisted local recordings and revision-safe human reference storage."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import threading
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabConflict, LabInputError
from overlap_lab.models import ReferenceAnnotation


class FileStudyRepository:
    def __init__(self, current: Path, uploads: Path, storage: Path, config: LabConfig) -> None:
        self.current, self.uploads, self.storage, self.config = current.resolve(), uploads.resolve(), storage.resolve(), config
        self._lock = threading.Lock()
        storage.mkdir(parents=True, exist_ok=True)

    def folder(self, recording_id: str) -> Path:
        """Resolve opaque recording IDs, never a client-supplied filesystem path."""
        if recording_id == "current":
            folder = self.current
        elif isinstance(recording_id, str) and re.fullmatch(r"[a-f0-9]{32}", recording_id):
            folder = (self.uploads / recording_id / "result").resolve()
            if not folder.is_relative_to(self.uploads):
                raise LabInputError("Recording path escapes upload storage")
        else:
            raise LabInputError("Unknown recording")
        if not (folder / "transcript.json").is_file() or not (folder / "processing.wav").is_file():
            raise LabInputError("This recording is not ready for experiments")
        return folder

    def list_recordings(self) -> list[dict[str, object]]:
        ids = ["current"]
        if self.uploads.is_dir():
            ids += [p.name for p in sorted(self.uploads.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    if p.is_dir() and re.fullmatch(r"[a-f0-9]{32}", p.name)]
        result: list[dict[str, object]] = []
        for recording_id in ids:
            try:
                data = json.loads((self.folder(recording_id) / "transcript.json").read_text(encoding="utf-8"))
                result.append({"id": recording_id, "filename": data["filename"]})
            except (LabInputError, OSError, ValueError, KeyError):
                continue
        return result

    def study(self, recording_id: str) -> dict[str, object]:
        """Return presentation data only; private paths/configs never leave the adapter."""
        try:
            data = json.loads((self.folder(recording_id) / "transcript.json").read_text(encoding="utf-8"))
            regions = data["variants"]["nemo_native"]["turns"]
            speakers = sorted({r["speaker"] for r in regions if r["speaker"] != "UNKNOWN"})
            regions = [{k: r[k] for k in ("id", "speaker", "start", "end")} for r in regions if r["speaker"] in speakers]
            decisions = data.get("recovery", {}).get("decisions", [])
            if not decisions:
                decisions = [{"start": max(a["start"], b["start"]), "end": min(a["end"], b["end"]),
                              "status": "unreviewed", "reason": "detected_overlap"}
                             for i, a in enumerate(regions) for b in regions[i+1:]
                             if a["speaker"] != b["speaker"] and min(a["end"], b["end"]) > max(a["start"], b["start"])]
            clips = []
            for i, row in enumerate(decisions):
                active = sorted({r["speaker"] for r in regions if min(r["end"], row["end"]) > max(r["start"], row["start"])})
                if len(active) == 2:
                    clips.append({"id": f"overlap_{i:04d}", "start": row["start"], "end": row["end"],
                                  "speakers": active, "status": row["status"], "reason": row["reason"]})
            audio_name = data["audio"]
            if not re.fullmatch(r"audio\.(mp3|wav|m4a|mp4|flac|ogg|webm)", audio_name):
                raise LabInputError("Invalid recording audio asset")
            if not isinstance(data["source_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", data["source_sha256"]):
                raise LabInputError("Invalid recording source identity")
            return {"id": recording_id, "filename": data["filename"], "source_hash": data["source_sha256"],
                    "duration": data["duration"], "waveform": data["waveform"], "regions": regions, "speakers": speakers,
                    "audio_url": f"/{audio_name}" if recording_id == "current" else f"/recordings/{recording_id}/{audio_name}",
                    "clips": clips, "existing_variants": data["variants"]}
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise LabInputError("Recording data is incomplete; rebuild its listening page") from error

    def annotations(self, recording_id: str) -> dict[str, object]:
        study = self.study(recording_id)
        path = self.storage / f"{recording_id}-{study['source_hash']}.json"
        with self._lock:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["source_hash"] == study["source_hash"]:
                    return data
        return {"recording_id": recording_id, "source_hash": study["source_hash"], "revision": 0, "items": {}}

    def save_reference(self, recording_id: str, source_hash: str, revision: int,
                       reference: ReferenceAnnotation) -> dict[str, object]:
        """Atomic CAS save prevents another browser tab silently losing edits."""
        if self.study(recording_id)["source_hash"] != source_hash:
            raise LabConflict("This recording changed. Reload before saving references.")
        path = self.storage / f"{recording_id}-{source_hash}.json"
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
                "recording_id": recording_id, "source_hash": source_hash, "revision": 0, "items": {}}
            if data["source_hash"] != source_hash or data["revision"] != revision:
                raise LabConflict("Another tab saved newer references. Reload to avoid overwriting them.")
            if reference.clip.id not in data["items"] and len(data["items"]) >= self.config.max_annotations:
                raise LabInputError("Reference limit reached for this recording")
            data["items"][reference.clip.id] = asdict(reference)
            data["revision"] += 1
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            return data
