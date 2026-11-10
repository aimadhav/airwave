"""Uploads, bounded jobs, private storage and HTTP behavior without real models."""

from dataclasses import replace
from functools import partial
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from typing import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from recording_viewer.adapters.mock_upload_processor import MockRecordingProcessor
from recording_viewer.adapters.upload_store import FileRecordingUploadStore
from recording_viewer.exceptions import UploadInputError, UploadQueueFull
from recording_viewer.models.upload import StoredRecording
from recording_viewer.services.upload_service import UploadService
from recording_viewer.upload_config import UploadConfig
from scripts.serve_recording_viewer import RecordingViewerHandler


def configuration() -> UploadConfig:
    """Load versioned local policy, not a remote dependency."""
    return UploadConfig.load(Path("configs/recording_upload.json"))


def test_upload_preserves_names_and_isolates_results(tmp_path: Path) -> None:
    service = UploadService(FileRecordingUploadStore(tmp_path), MockRecordingProcessor(), configuration())
    first = service.submit("meeting.wav", io.BytesIO(b"test one"), 8)
    second = service.submit("meeting.wav", io.BytesIO(b"test two"), 8)
    service.close()
    assert first.id != second.id
    assert (tmp_path / first.id / "meeting.wav").read_bytes() == b"test one"
    assert (tmp_path / second.id / "meeting.wav").read_bytes() == b"test two"
    assert service.get(first.id).status == "completed"
    assert service.get(first.id).result_url == f"/recordings/{first.id}/index.html"


@pytest.mark.parametrize("name", ["../a.wav", "C:\\a.wav", "a/b.wav", "NUL.wav", "bad.txt", "bad\x00.wav", "bad.wav "])
def test_bad_filenames_rejected_before_writing(tmp_path: Path, name: str) -> None:
    service = UploadService(FileRecordingUploadStore(tmp_path), MockRecordingProcessor(), configuration())
    try:
        with pytest.raises(UploadInputError):
            service.submit(name, io.BytesIO(b"x"), 1)
        assert list(tmp_path.iterdir()) == []
    finally:
        service.close()


def test_size_limits_and_interrupted_upload(tmp_path: Path) -> None:
    service = UploadService(FileRecordingUploadStore(tmp_path), MockRecordingProcessor(), replace(configuration(), max_upload_bytes=5))
    try:
        for length in (0, 6):
            with pytest.raises(UploadInputError):
                service.submit("a.wav", io.BytesIO(b"x"), length)
        with pytest.raises(UploadInputError, match="interrupted"):
            service.submit("a.wav", io.BytesIO(b"x"), 3)
        assert not list(tmp_path.glob("*/result/index.html"))
        assert not list(tmp_path.glob("*/a.wav"))
    finally:
        service.close()


def test_bounded_queue_and_explicit_failure(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()

    class FailingProcessor:
        def process(self, job_id: str, recording: StoredRecording) -> None:
            entered.set()
            release.wait(5)
            raise RuntimeError("private provider details must not leak")

    service = UploadService(FileRecordingUploadStore(tmp_path), FailingProcessor(), replace(configuration(), max_pending_jobs=1))
    job = service.submit("a.wav", io.BytesIO(b"x"), 1)
    assert entered.wait(5)
    try:
        with pytest.raises(UploadQueueFull):
            service.submit("b.wav", io.BytesIO(b"x"), 1)
    finally:
        release.set()
        service.close()
    assert service.get(job.id).status == "failed"
    assert service.get(job.id).result_url is None
    assert "private provider" not in service.get(job.id).message


@pytest.fixture
def upload_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    service = UploadService(FileRecordingUploadStore(tmp_path / "uploads"), MockRecordingProcessor(), configuration())
    handler = partial(RecordingViewerHandler, directory=str(tmp_path), uploads=service,
                      upload_config=configuration(), upload_root=tmp_path / "uploads")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        service.close()


def test_http_upload_status_and_result(upload_server: tuple[str, Path]) -> None:
    base, root = upload_server
    with urlopen(base + "/api/upload-config") as response:
        assert json.load(response)["enabled"] is True
    request = Request(base + "/api/uploads", data=b"fixture", headers={"Content-Type": "application/octet-stream", "X-Filename": "example.wav"})
    with urlopen(request) as response:
        assert response.status == 202
        job = json.load(response)
    for _ in range(50):
        with urlopen(base + "/api/jobs/" + job["id"]) as response:
            status = json.load(response)
        if status["status"] == "completed":
            break
        time.sleep(.01)
    assert status["status"] == "completed"
    with urlopen(base + status["result_url"]) as response:
        assert b"Fixture only" in response.read()
    for route in (f"/recordings/{job['id']}/processing.log", "/recordings/../index.html", "/api/jobs/unknown"):
        with pytest.raises(HTTPError) as error:
            urlopen(base + route)
        assert error.value.code == 404


@pytest.mark.parametrize("headers,status", [({"Origin": "https://evil.example"}, 403),
    ({"Origin": "null"}, 403), ({"Host": "evil.example"}, 403),
    ({"Content-Type": "text/plain"}, 415), ({"Content-Length": "9999999999"}, 413)])
def test_unsafe_http_upload_rejected(upload_server: tuple[str, Path], headers: dict[str, str], status: int) -> None:
    base, root = upload_server
    values = {"Content-Type": "application/octet-stream", "X-Filename": "file.wav", **headers}
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/api/uploads", data=b"bad", headers=values))
    assert error.value.code == status
    assert not (root / "uploads").exists()
