import base64
import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path
import pytest

from app import AudioFileAppHandler
from ai_layer.audio_ingestion.sample_generator import generate_synthetic_acoustic_wav


@pytest.fixture(scope="module")
def test_server(tmp_path_factory):
    # HTTP regression tests must not fill the user's workspace database.
    import app
    test_dir = tmp_path_factory.mktemp("web_api")
    config_path = test_dir / "config.yaml"
    config_text = Path("configs/ai_layer_config.yaml").read_text(encoding="utf-8")
    config_path.write_text(config_text.replace('sqlite_path: "data/ai_layer.db"',
                           f'sqlite_path: "{(test_dir / "test.db").as_posix()}"'), encoding="utf-8")
    original_path = app.CONFIG_PATH
    app.CONFIG_PATH = str(config_path)
    server = HTTPServer(("127.0.0.1", 0), AudioFileAppHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()
    app.CONFIG_PATH = original_path


def test_api_status_endpoint(test_server):
    req = urllib.request.Request(f"{test_server}/api/status")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "mode" in data
        assert "sessions_count" in data


def test_offline_demo_timeline_and_clip(test_server: str) -> None:
    """The dashboard gets actual clip intervals and explicit fixture provenance."""
    req = urllib.request.Request(
        f"{test_server}/api/demo-multispeaker", data=b'{"mode":"openai"}',
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
    assert result["dsp_segments"] == 4
    assert len(result["speaker_ids"]) == 3
    with urllib.request.urlopen(f"{test_server}/api/sessions/{result['session_id']}") as response:
        details = json.loads(response.read())
    assert details["dsp"]["fixture"] is True
    assert details["dsp"]["analysis_mode"] == "mock"
    assert details["facts"][0]["statement"] in [t["text"] for t in details["transcript"]]
    assert "contract" not in details["facts"][0]["statement"].lower()
    assert details["facts"][0]["source_turn_id"] == details["transcript"][1]["turn_id"]
    segment = details["dsp"]["segments"][0]
    with urllib.request.urlopen(f"{test_server}/api/audio/{result['session_id']}/{segment['segment_id']}") as response:
        assert response.headers["Content-Type"].startswith("audio/wav")
        assert response.read(4) == b"RIFF"


def test_unknown_session_and_static_traversal_are_rejected(test_server: str) -> None:
    for path in ("/api/sessions/not_a_session", "/web/%2e%2e/app.py", "/api/audio/x/../app"):
        try:
            urllib.request.urlopen(test_server + path)
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError(f"Unsafe or unknown resource was served: {path}")
    with urllib.request.urlopen(test_server + "/web/workspace.css") as response:
        assert response.headers["Content-Type"].startswith("text/css")


def test_invalid_json_is_not_silently_treated_as_demo_request(test_server: str) -> None:
    for body in (b"not-json", b"[]", b"null"):
        req = urllib.request.Request(test_server + "/api/demo-multispeaker", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert "error" in json.loads(error.read())
        else:
            raise AssertionError("Invalid request unexpectedly started a demo")


def test_api_upload_base64_audio(test_server, tmp_path):
    # Generate small synthetic test wave
    sample_wav = tmp_path / "test_upload.wav"
    generate_synthetic_acoustic_wav(sample_wav)
    raw_bytes = sample_wav.read_bytes()
    b64_str = f"data:audio/wav;base64,{base64.b64encode(raw_bytes).decode('ascii')}"

    payload = {
        "audio_base64": b64_str,
        "filename": "meeting_test_upload.wav",
        "title": "Uploaded Voice Meeting Test",
        "mode": "mock",
        "sync": True,
    }

    req = urllib.request.Request(
        f"{test_server}/api/upload",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "success"
        assert "session_id" in res
        assert res["title"] == "Uploaded Voice Meeting Test"
        assert res["turns_count"] > 0
        # Arbitrary WAVs in mock mode contain placeholder text, not evidence.
        assert res["facts_count"] == 0


def test_api_upload_async(test_server, tmp_path):
    sample_wav = tmp_path / "test_upload_async.wav"
    generate_synthetic_acoustic_wav(sample_wav)
    raw_bytes = sample_wav.read_bytes()
    b64_str = f"data:audio/wav;base64,{base64.b64encode(raw_bytes).decode('ascii')}"

    payload = {
        "audio_base64": b64_str,
        "filename": "async_meeting.wav",
        "title": "Async Processing Test",
        "mode": "mock",
        "sync": False,
    }

    req = urllib.request.Request(
        f"{test_server}/api/upload",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "processing"
        assert "session_id" in res
        assert res["title"] == "Async Processing Test"


def test_api_upload_missing_audio(test_server):
    payload = {
        "title": "Missing Audio",
        "mode": "mock",
    }

    req = urllib.request.Request(
        f"{test_server}/api/upload",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)

    assert exc_info.value.code == 400


def test_api_fetch_audio_url(test_server, tmp_path):
    # Create local file and serve it via test_server static file or file:// URL
    test_wav = tmp_path / "remote_meeting.wav"
    generate_synthetic_acoustic_wav(test_wav)
    file_url = test_wav.as_uri()

    payload = {
        "url": file_url,
        "title": "Fetched Remote Meeting",
        "mode": "mock",
        "sync": True,
    }

    req = urllib.request.Request(
        f"{test_server}/api/fetch-audio",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "success"
        assert "session_id" in res
        assert res["title"] == "Fetched Remote Meeting"
        assert res["turns_count"] > 0
