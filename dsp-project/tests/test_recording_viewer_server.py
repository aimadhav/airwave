"""Private local serving and timestamp-seeking regression tests."""

from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from collections.abc import Iterator
import pytest
from scripts.serve_recording_viewer import RecordingViewerHandler


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    (tmp_path / "audio.mp3").write_bytes(bytes(range(100)))
    (tmp_path / "provenance.json").write_text("private processing metadata", encoding="utf-8")
    http = ThreadingHTTPServer(("127.0.0.1", 0), partial(RecordingViewerHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize("byte_range,expected", [("bytes=10-19", bytes(range(10, 20))),
    ("bytes=95-", bytes(range(95, 100))), ("bytes=-5", bytes(range(95, 100)))])
def test_audio_ranges(server: str, byte_range: str, expected: bytes) -> None:
    with urlopen(Request(server + "/audio.mp3", headers={"Range": byte_range})) as response:
        assert response.status == 206 and response.headers["Accept-Ranges"] == "bytes"
        assert response.read() == expected


def test_full_audio_and_head(server: str) -> None:
    with urlopen(server + "/audio.mp3") as response:
        assert response.status == 200 and response.read() == bytes(range(100))
    with urlopen(Request(server + "/audio.mp3", method="HEAD")) as response:
        assert response.headers["Content-Length"] == "100" and response.read() == b""


@pytest.mark.parametrize("path,status", [("/provenance.json", 404), ("/%2e%2e/audio.mp3", 404)])
def test_private_files_and_traversal_are_not_served(server: str, path: str, status: int) -> None:
    with pytest.raises(HTTPError) as error:
        urlopen(server + path)
    assert error.value.code == status


@pytest.mark.parametrize("byte_range", ["bytes=100-", "bytes=5-3", "bytes=-0", "bytes=0-1,3-4"])
def test_invalid_range(server: str, byte_range: str) -> None:
    with pytest.raises(HTTPError) as error:
        urlopen(Request(server + "/audio.mp3", headers={"Range": byte_range}))
    assert error.value.code == 416
