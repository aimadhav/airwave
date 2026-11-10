"""Loopback-only private viewer with byte-range audio seeking."""

import argparse
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import json
from pathlib import Path
import re
import sys
import webbrowser
from typing import BinaryIO
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recording_viewer.exceptions import RecordingViewerError, UploadInputError, UploadQueueFull
from recording_viewer.services.upload_service import UploadService
from recording_viewer.upload_config import UploadConfig
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabBusy, LabConflict, LabError, LabInputError
from overlap_lab.models import ExperimentRequest, ReferenceAnnotation
from overlap_lab.models.listening import ListeningReview
from overlap_lab.services.listening_service import ListeningReviewService
from overlap_lab.services.lab_service import LabService
from overlap_lab.services.scoring import score_speakers, tokenize


class RecordingViewerHandler(SimpleHTTPRequestHandler):
    """Serve only presentation assets; never expose processing/provenance files."""

    ALLOWED = {"index.html", "viewer.css", "viewer.js", "upload.css", "upload.js", "audio.mp3", "audio.wav",
               "audio.m4a", "audio.mp4", "audio.flac", "audio.ogg", "audio.webm", "transcript.json", "overlap-preview.jpg"}

    def __init__(self, *args: object, uploads: UploadService | None = None,
                 upload_config: UploadConfig | None = None, upload_root: Path | None = None,
                 lab: LabService | None = None, lab_config: LabConfig | None = None,
                 listening: ListeningReviewService | None = None, **kwargs: object) -> None:
        self.uploads, self.upload_config, self.upload_root = uploads, upload_config, upload_root
        self.lab, self.lab_config = lab, lab_config
        self.listening = listening
        super().__init__(*args, **kwargs)

    def _same_origin(self) -> bool:
        """Reject cross-site writes and DNS-rebinding hostnames; no CORS wildcard."""
        host = self.headers.get("Host", "")
        try:
            location = urlsplit("http://" + host)
            valid = (location.hostname in {"127.0.0.1", "localhost"} and location.port == self.server.server_port
                     and not any((location.path, location.query, location.fragment, location.username, location.password)))
        except ValueError:
            return False
        origin = self.headers.get("Origin")
        return valid and (origin is None or origin == "http://" + host)

    def _json(self, value: object, status: int = 200) -> None:
        """Return small same-origin job responses without exposing logs."""
        # Drain only small, rejected bodies. Closing a Windows socket with unread
        # bytes can reset it before the client receives the useful error status.
        if status >= 400 and self.command == "POST" and not getattr(self, "_body_handled", False):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if 0 < length <= 64 * 1024:
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Expose upload policy and immutable processing status snapshots."""
        if not self._same_origin():
            return self._json({"error": "Only this local app may access the recording service"}, 403)
        path = urlsplit(self.path).path
        if path.startswith("/api/lab/"):
            return self._lab_get(path)
        if path == "/api/upload-config":
            config = self.upload_config
            return self._json({"enabled": self.uploads is not None, "max_bytes": config.max_upload_bytes if config else 0,
                "max_audio_seconds": config.max_audio_seconds if config else 0,
                "extensions": config.allowed_extensions if config else [], "poll_interval_ms": config.poll_interval_ms if config else 0})
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            job = self.uploads.get(job_id) if self.uploads and re.fullmatch(r"[a-f0-9]{32}", job_id) else None
            return self._json(asdict(job), 200) if job else self._json({"error": "Recording job not found"}, 404)
        return super().do_GET()

    def do_HEAD(self) -> None:
        """Apply the same host boundary to metadata-only media requests."""
        if not self._same_origin():
            self.send_error(403)
            return
        return super().do_HEAD()

    def do_POST(self) -> None:
        """Stream a bounded binary upload; never accept a client filesystem path."""
        if self.upload_config:
            self.connection.settimeout(self.upload_config.request_timeout_seconds)
        if not self._same_origin():
            return self._json({"error": "Cross-origin uploads are not allowed"}, 403)
        if urlsplit(self.path).path.startswith("/api/lab/"):
            return self._lab_post(urlsplit(self.path).path)
        if self.path != "/api/uploads" or self.uploads is None or self.upload_config is None:
            return self._json({"error": "Local upload processing is not enabled"}, 404)
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/octet-stream":
            return self._json({"error": "Send the recording as a binary upload"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > self.upload_config.max_upload_bytes:
                return self._json({"error": "Recording exceeds the upload size limit"}, 413)
            self.connection.settimeout(self.upload_config.request_timeout_seconds)
            filename = unquote(self.headers.get("X-Filename", ""))
            self.uploads.validate(filename, length)
            self._body_handled = True
            job = self.uploads.submit(filename, self.rfile, length)
            return self._json(asdict(job), 202)
        except UploadQueueFull as error:
            return self._json({"error": str(error)}, 429)
        except (UploadInputError, ValueError) as error:
            return self._json({"error": str(error)}, 400)
        except (OSError, RecordingViewerError):
            return self._json({"error": "Upload could not be saved. Check the file and local storage."}, 400)

    def _lab_get(self, path: str) -> None:
        """Expose allowlisted studies and job DTOs, never arbitrary files."""
        if self.lab is None or self.lab_config is None:
            return self._json({"error": "Evaluation lab is not enabled"}, 404)
        try:
            if path == "/api/lab/listening" and self.listening:
                recording_id = parse_qs(urlsplit(self.path).query).get("recording", ["current"])[0]
                return self._json(self.listening.bootstrap(recording_id))
            if path == "/api/lab/bootstrap":
                recording_id = parse_qs(urlsplit(self.path).query).get("recording", ["current"])[0]
                return self._json({"recordings": self.lab.repository.list_recordings(),
                    "study": self.lab.repository.study(recording_id), "annotations": self.lab.repository.annotations(recording_id),
                    "variants": self.lab.processor.capabilities(), "jobs": [asdict(j) for j in self.lab.history(recording_id)],
                    "limits": {k: getattr(self.lab_config, k) for k in ("max_clip_seconds", "max_context_seconds",
                        "default_context_seconds", "poll_interval_ms", "max_reference_characters")}})
            if path.startswith("/api/lab/jobs/"):
                job_id = path.removeprefix("/api/lab/jobs/")
                job = self.lab.get(job_id) if re.fullmatch(r"[a-f0-9]{32}", job_id) else None
                return self._json(asdict(job), 200) if job else self._json({"error": "Experiment not found"}, 404)
            return self._json({"error": "Unknown lab endpoint"}, 404)
        except LabError as error:
            return self._json({"error": str(error)}, 400)
        except (OSError, ValueError, KeyError, TypeError):
            return self._json({"error": "Lab data could not be read"}, 400)

    def _lab_post(self, path: str) -> None:
        """Bounded JSON requests; no executable paths or model downloads."""
        if self.lab is None or self.lab_config is None:
            return self._json({"error": "Evaluation lab is not enabled"}, 404)
        try:
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self._json({"error": "Send application/json"}, 415)
            if self.headers.get("Transfer-Encoding"):
                return self._json({"error": "Chunked JSON is not supported"}, 400)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= self.lab_config.request_limit_bytes:
                return self._json({"error": "Lab request is empty or too large"}, 413)
            self.connection.settimeout(self.lab_config.request_timeout_seconds)
            self._body_handled = True
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise LabInputError("Incomplete JSON request")
            values = json.loads(raw)
            if not isinstance(values, dict):
                raise LabInputError("Expected a JSON object")
            if path == "/api/lab/listening-reviews" and self.listening:
                return self._json(self.listening.save(values["recording_id"], values["source_hash"], values["revision"],
                                                       ListeningReview.from_mapping(values["review"])))
            if path == "/api/lab/references":
                reference = ReferenceAnnotation.from_mapping(values["reference"])
                self.lab.validate_reference(values["recording_id"], reference)
                revision = values["revision"]
                if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                    raise LabInputError("Invalid reference revision")
                return self._json(self.lab.repository.save_reference(values["recording_id"], values["source_hash"], revision, reference))
            if path == "/api/lab/runs":
                return self._json(asdict(self.lab.submit(ExperimentRequest.from_mapping(values))), 202)
            if path == "/api/lab/score":
                reference = ReferenceAnnotation.from_mapping(values["reference"])
                self.lab.validate_reference(values["recording_id"], reference)
                hypotheses = values["hypotheses"]
                if not isinstance(hypotheses, dict) or set(hypotheses) != set(reference.texts) or any(
                    not isinstance(t, str) or len(t) > self.lab_config.max_reference_characters
                    or len(tokenize(t)) > self.lab_config.max_reference_words for t in hypotheses.values()):
                    raise LabInputError("Invalid speaker hypotheses")
                if not reference.reviewed:
                    raise LabInputError("Review the reference before computing WER")
                return self._json({"scores": score_speakers(reference, hypotheses), "attribution": "manual_for_evaluation_only"})
            return self._json({"error": "Unknown lab endpoint"}, 404)
        except LabConflict as error:
            return self._json({"error": str(error)}, 409)
        except LabBusy as error:
            return self._json({"error": str(error)}, 429)
        except (LabInputError, ValueError, TypeError, KeyError, OSError) as error:
            return self._json({"error": str(error) if isinstance(error, LabError) else "Invalid lab request"}, 400)

    def send_head(self) -> BinaryIO | None:
        """Honor one byte range so clicking a timestamp actually seeks audio."""
        name = unquote(urlsplit(self.path).path).removeprefix("/") or "index.html"
        root = Path(self.directory).resolve()
        if name in {"lab/listen", "lab/listen/", "lab/listen/index.html", "lab/listen/listen.js", "lab/listen/listen.css"}:
            root = ROOT / "web/overlap-lab/listen"
            name = "index.html" if name in {"lab/listen", "lab/listen/"} else name.split("/")[-1]
        elif name in {"lab", "lab/", "lab/index.html", "lab/lab.css", "lab/lab.js"}:
            root = ROOT / "web/overlap-lab"
            name = "index.html" if name in {"lab", "lab/"} else name.split("/")[-1]
        elif name.startswith("lab-audio/"):
            parts = name.split("/")
            if (not self.lab or not self.lab_config or len(parts) != 3 or not re.fullmatch(r"[a-f0-9]{32}", parts[1])
                    or not re.fullmatch(r"mixture\.wav|sepformer_track_[01]\.wav", parts[2])
                    or not self.lab.get(parts[1]) or self.lab.get(parts[1]).status != "completed"):
                self.send_error(404)
                return None
            root = (ROOT / self.lab_config.storage_directory / "runs" / parts[1]).resolve()
            name = parts[2]
        else:
            root = Path(self.directory).resolve()
        if name.startswith("recordings/"):
            parts = name.split("/")
            if self.upload_root is None or len(parts) != 3 or not re.fullmatch(r"[a-f0-9]{32}", parts[1]):
                self.send_error(404)
                return None
            root = (self.upload_root / parts[1] / "result").resolve()
            if not root.is_relative_to(self.upload_root.resolve()):
                self.send_error(404)
                return None
            name = parts[2]
        is_lab = urlsplit(self.path).path.startswith(("/lab/", "/lab-audio/")) or urlsplit(self.path).path == "/lab"
        if name not in self.ALLOWED and not (is_lab and name in {"index.html", "lab.css", "lab.js", "listen.css", "listen.js", "mixture.wav", "sepformer_track_0.wav", "sepformer_track_1.wav"}):
            self.send_error(404)
            return None
        target = (root / name).resolve()
        if target.parent != root or not target.is_file():
            self.send_error(404)
            return None
        size = target.stat().st_size
        start, end = 0, size - 1
        requested_range = self.headers.get("Range")
        if requested_range:
            try:
                unit, value = requested_range.split("=", 1)
                left, right = value.split("-", 1)
                if unit != "bytes" or "," in value:
                    raise ValueError("Only one byte range is supported")
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    suffix = int(right)
                    if suffix <= 0:
                        raise ValueError("Invalid suffix")
                    start = max(0, size - suffix)
                if start < 0 or start >= size or end < start:
                    raise ValueError("Unsatisfiable range")
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
        source = target.open("rb")
        source.seek(start)
        self._remaining = max(0, end - start + 1)
        self.send_response(206 if requested_range else 200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-cache")
        if requested_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        return source

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        """Stream exactly the requested range, not the whole MP3 on every seek."""
        remaining = self._remaining
        try:
            while remaining:
                block = source.read(min(64 * 1024, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # Browsers routinely abort a media request before seeking.


def main() -> None:
    """Bind to loopback only; require explicit preview folder and port."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--open-browser", action="store_true", help="Open the local app after it binds successfully")
    parser.add_argument("--upload-config", type=Path, default=ROOT / "configs/recording_upload.json")
    args = parser.parse_args()
    # Composition root owns concrete adapters; job logic uses their protocols.
    from recording_viewer.adapters.upload_store import FileRecordingUploadStore
    from recording_viewer.adapters.upload_processor import SubprocessRecordingProcessor
    from recording_viewer.adapters.landing_page import create_upload_landing
    config = UploadConfig.load(args.upload_config)
    directory = args.directory or ROOT / json.loads((ROOT / config.base_config).read_text(encoding="utf-8"))["output_directory"]
    port = args.port if args.port is not None else urlsplit(config.local_app_url).port
    if port is None:
        parser.error("Configure a valid local port")
    try:
        create_upload_landing(directory, ROOT / "web/recording-viewer", config.local_app_url)
    except RecordingViewerError as error:
        parser.error(str(error))
    upload_root = (ROOT / config.upload_directory).resolve()
    if not upload_root.is_relative_to((ROOT / "data/listening-room").resolve()):
        parser.error("Upload storage must stay inside the private listening-room directory")
    uploads = UploadService(FileRecordingUploadStore(upload_root), SubprocessRecordingProcessor(ROOT, config), config)
    from overlap_lab.adapters.file_studies import FileStudyRepository
    from overlap_lab.adapters.local_experiments import LocalExperimentProcessor
    from overlap_lab.adapters.listening_store import FileListeningReviewRepository
    lab_config = LabConfig.load(ROOT / "configs/overlap_lab.json")
    lab_storage = (ROOT / lab_config.storage_directory).resolve()
    if not lab_storage.is_relative_to((ROOT / "data").resolve()):
        parser.error("Lab data must stay inside the private data directory")
    repository = FileStudyRepository(directory, upload_root, lab_storage / "references", lab_config)
    listening = ListeningReviewService(repository,
        FileListeningReviewRepository(repository, lab_storage / "listening-reviews", lab_config), lab_config)
    lab = LabService(repository, LocalExperimentProcessor(ROOT, directory, upload_root, repository, lab_config), lab_config)
    try:
        with ThreadingHTTPServer(("127.0.0.1", port), partial(RecordingViewerHandler, directory=str(directory),
                    uploads=uploads, upload_config=config, upload_root=upload_root, lab=lab, lab_config=lab_config,
                    listening=listening)) as server:
            print(f"Private listening room: http://127.0.0.1:{port}/", flush=True)
            if args.open_browser:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            server.serve_forever()
    finally:
        uploads.close()
        lab.close()


if __name__ == "__main__":
    main()
