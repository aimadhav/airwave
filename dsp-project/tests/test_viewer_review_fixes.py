"""Regression coverage for review findings; no models, network APIs or user data."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from recording_viewer.adapters.landing_page import create_upload_landing
from recording_viewer.config import RecordingViewerConfig
from recording_viewer.exceptions import RecordingViewerError


ROOT = Path(__file__).resolve().parents[1]


def test_first_run_landing_needs_no_audio_or_models(tmp_path: Path) -> None:
    folder = tmp_path / "empty-listening-room"
    create_upload_landing(folder, ROOT / "web/recording-viewer", "http://127.0.0.1:5095/")
    page = (folder / "index.html").read_text(encoding="utf-8")
    assert "No recording has been processed yet" in page and 'id="new-recording"' in page
    assert "__RECORDING_DATA__" not in page
    assert not (folder / "transcript.json").exists()
    for asset in ("viewer.css", "upload.css", "upload.js"):
        assert (folder / asset).is_file()


def test_landing_never_overwrites_existing_recording_or_assets(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("existing user page", encoding="utf-8")
    (tmp_path / "viewer.css").write_text("existing user css", encoding="utf-8")
    create_upload_landing(tmp_path, ROOT / "web/recording-viewer", "http://localhost:5095/")
    assert (tmp_path / "index.html").read_text() == "existing user page"
    assert (tmp_path / "viewer.css").read_text() == "existing user css"


def test_landing_escapes_script_data_and_translates_io_errors(tmp_path: Path) -> None:
    folder = tmp_path / "landing"
    create_upload_landing(folder, ROOT / "web/recording-viewer", "http://localhost/?x=</script>&")
    assert "?x=</script>" not in (folder / "index.html").read_text()
    with pytest.raises(RecordingViewerError):
        create_upload_landing(tmp_path / "missing", tmp_path / "absent-assets", "http://localhost:5095/")


def test_local_paths_override_portable_configuration_only_on_this_machine(tmp_path: Path) -> None:
    values = asdict(RecordingViewerConfig.load(ROOT / "configs/recording_viewer.json"))
    values["source"] = "data/recordings/input.wav"
    policy = tmp_path / "viewer.json"
    policy.write_text(json.dumps(values), encoding="utf-8")
    assert RecordingViewerConfig.load(policy).source == "data/recordings/input.wav"
    policy.with_suffix(".local.json").write_text(json.dumps({"source": "local-only-recording.wav"}), encoding="utf-8")
    assert RecordingViewerConfig.load(policy).source == "local-only-recording.wav"
    assert json.loads(policy.read_text())["source"] == "data/recordings/input.wav"


@pytest.mark.parametrize("invalid", ["broken JSON", "[]", '{"unknown_setting": true}'])
def test_invalid_local_override_is_a_typed_configuration_error(tmp_path: Path, invalid: str) -> None:
    policy = tmp_path / "viewer.json"
    values = asdict(RecordingViewerConfig.load(ROOT / "configs/recording_viewer.json"))
    policy.write_text(json.dumps(values), encoding="utf-8")
    policy.with_suffix(".local.json").write_text(invalid, encoding="utf-8")
    with pytest.raises(RecordingViewerError):
        RecordingViewerConfig.load(policy)
