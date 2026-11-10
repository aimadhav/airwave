"""Hosted website preparation uses local assets and never re-runs speech models."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import pytest

from recording_viewer.exceptions import RecordingViewerError
from scripts.prepare_podcast_site import PodcastSiteConfig, prepare

ROOT = Path(__file__).resolve().parents[1]


def policy() -> PodcastSiteConfig:
    return PodcastSiteConfig.load(ROOT / "configs/podcast_site.json")


def test_hosted_copy_has_no_private_github_or_localhost_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = policy()
    source, site = tmp_path / config.demo_directory, tmp_path / config.site_directory
    source.mkdir(parents=True);(site / ".openai").mkdir(parents=True)
    (site / ".openai/hosting.json").write_text(json.dumps({"project_id": "synthetic-site-id", "static": {"directory": "dist"}}))
    document = {"audio": "audio.mp3", "read_only": True, "source_sha256": "a"*64, "notice": "Synthetic fixture."}
    (source / "transcript.json").write_text(json.dumps(document))
    (source / "results.json").write_text("{}")
    (source / "audio.mp3").write_bytes(b"synthetic source")
    (source / "examples").mkdir();(source / "examples/example.wav").write_bytes(b"unchanged fixture")
    for name in ("viewer.css", "viewer.js", "upload.css", "upload.js", "demo-examples.js", "TRANSCRIPT.md"):
        (source / name).write_text("fixture", encoding="utf-8")
    templates = tmp_path / "web/recording-viewer"
    templates.mkdir(parents=True)
    for name in ("viewer.js", "buffered-playback.js"):
        (templates / name).write_text((ROOT / "web/recording-viewer" / name).read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("pipeline.html", "pipeline.js", "pipeline.css"):
        (templates / name).write_text((ROOT / "web/recording-viewer" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (templates / "pipeline-audit.json").write_text(json.dumps({"source_sha256": "a"*64}))
    for name in ("index.html", "demo-examples.html"):
        (templates / name).write_text('<strong>Original recording</strong><script>__RECORDING_DATA__</script><a href="audio.mp3">Download full MP3</a>')
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "synthetic-ffmpeg"))

    def fake_encode(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        assert command[command.index("-c:a")+1] == "aac"
        assert command[command.index("-b:a")+1] == config.audio_bit_rate
        assert options["shell"] is False
        Path(command[-1]).write_bytes(b"synthetic web copy")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("scripts.prepare_podcast_site.subprocess.run", fake_encode)
    output = prepare(tmp_path, config)
    hosted = json.loads((output / "transcript.json").read_text(encoding="utf-8"))
    assert hosted["audio"] == config.audio_filename and hosted["read_only"] is True
    assert hosted["web_playback"]["buffer_before_seek"] is True
    assert (output / "buffered-playback.js").is_file()
    assert (output / "pipeline.html").is_file()
    assert "__PIPELINE_AUDIT__" not in (output / "pipeline.html").read_text(encoding="utf-8")
    assert hosted["source_sha256"] == hosted["web_playback"]["original_sha256"] == "a"*64
    assert json.loads((source / "transcript.json").read_text())["audio"] == "audio.mp3"
    assert not (output / "audio.mp3").exists()
    assert (output / "examples/example.wav").read_bytes() == b"unchanged fixture"
    assert "Web playback copy" in (output / "index.html").read_text(encoding="utf-8")
    assert "Download full MP3" not in (output / "examples.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("changes", [{"audio_sample_rate": 0}, {"audio_channels": True}, {"audio_filename": "../audio.m4a"},
    {"audio_bit_rate": "0k"}, {"audio_bit_rate": "56"}, {"encode_timeout_seconds": float("nan")}, {"max_asset_bytes": -1},
    {"playback_load_timeout_seconds": 0}])
def test_hosting_audio_policy_is_validated(changes: dict[str, object]) -> None:
    with pytest.raises(RecordingViewerError):
        replace(policy(), **changes)


def test_buffered_playback_with_mock_browser() -> None:
    """No network, browser, audio decoder or live hosting is required."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for isolated browser-adapter tests")
    subprocess.run([node, "--test", str(ROOT / "tests/browser/buffered-playback.test.cjs")],
                   check=True, capture_output=True, text=True, timeout=30)


def test_mismatched_audit_fails_before_writing_site_assets(tmp_path: Path) -> None:
    config = policy()
    source, site = tmp_path / config.demo_directory, tmp_path / config.site_directory
    source.mkdir(parents=True)
    (site / ".openai").mkdir(parents=True)
    (site / ".openai/hosting.json").write_text(json.dumps({"project_id": "fixture", "static": {"directory": "dist"}}))
    (source / "transcript.json").write_text(json.dumps({"source_sha256": "a"*64}))
    (source / "results.json").write_text("{}")
    templates = tmp_path / "web/recording-viewer"
    templates.mkdir(parents=True)
    (templates / "pipeline-audit.json").write_text(json.dumps({"source_sha256": "b"*64}))
    with pytest.raises(RecordingViewerError, match="different recording"):
        prepare(tmp_path, config)
    assert not (site / "dist").exists()
