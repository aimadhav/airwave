"""Portable demo exports are source-bound, read-only and free of private state."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import wave

import pytest

from recording_viewer.demo_config import DemoExample, PublicDemoConfig
from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.services.demo_snapshot import public_snapshot, script_json, transcript_markdown
from scripts.export_podcast_demo import export_demo


ROOT = Path(__file__).resolve().parents[1]


def document(source_hash: str = "a" * 64) -> dict[str, object]:
    """Explicitly synthetic metadata; never mistaken for a real model result."""
    return {"schema_version": 2, "filename": "synthetic-fixture.mp3", "source_sha256": source_hash,
        "duration": 3., "sample_rate": 16000, "audio": "audio.mp3", "recognizer": "fixture-only",
        "model_revision": "fixture", "word_count": 1, "waveform": [.1, .3], "notice": "Synthetic test only.",
        "configuration": {"source": "PRIVATE_ABSOLUTE_PATH"}, "local_app_url": "http://localhost:9999/",
        "variants": {"nemo_native": {"turns": [{"id": "a", "speaker": "SPEAKER_00", "start": 0., "end": 3.,
            "text": "Hello </script><script>fixture</script>", "uncertain": True, "overlap": True, "word_count": 1,
            "source_path": "PRIVATE_ABSOLUTE_PATH"}], "groups": [{"start": 0., "end": 3., "turn_ids": ["a"], "overlap": True}]},
            "private_debug_variant": {"raw_path": "PRIVATE_ABSOLUTE_PATH"}},
        "recovery": {"enabled": True, "windows": 1, "accepted": 0, "rejected": 1, "skipped": 0, "failed": 0,
                     "private_log": "PRIVATE_ABSOLUTE_PATH", "decisions": [
                         {"start": 1., "end": 2., "status": "rejected", "reason": "PRIVATE_ABSOLUTE_PATH"}]}}


def fixture(root: Path) -> PublicDemoConfig:
    source, feedback = root / "data/source", root / "data/feedback"
    source.mkdir(parents=True);feedback.mkdir(parents=True)
    audio = b"SYNTHETIC UNIT FIXTURE, NOT PLAYABLE MP3"
    (source / "audio.mp3").write_bytes(audio)
    source_hash = hashlib.sha256(audio).hexdigest()
    (source / "transcript.json").write_text(json.dumps(document(source_hash)), encoding="utf-8")
    clip = {"id": "overlap_0001", "start": 1., "end": 2., "speakers": ["SPEAKER_00", "SPEAKER_01"]}
    snapshot = {"reviews": {"source_hash": source_hash, "items": {clip["id"]: {"clip": clip}}}}
    (feedback / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    summary = [{"gain_mode": "shared_peak", "context_seconds": 1., "confirmed_overlaps_tested": 1,
        "confirmed_overlaps_accepted": 1, "negative_controls_tested": 0,
        "negative_controls_incorrectly_accepted": 0, "failed_runs": 0, "private_path": "PRIVATE_ABSOLUTE_PATH"}]
    (feedback / "recovery-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    job_id = "1"*32
    job = root / "data/overlap-lab/runs" / job_id
    job.mkdir(parents=True)
    result = {"source_hash": source_hash, "request": {"reference": {"clip": clip}, "context_seconds": 1.},
        "separator_gain_mode": "shared_peak", "crop_start": 0., "crop_end": 3., "private_path": "PRIVATE_ABSOLUTE_PATH",
        "variants": [{"id": "sepformer--base", "status": "accepted", "reason": "fixture_only", "tracks": [
            {"index": i, "text": "fixture words", "assigned_speaker": f"SPEAKER_{i:02d}", "private_path": "PRIVATE_ABSOLUTE_PATH"}
            for i in range(2)]}]}
    (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
    for filename in ("mixture.wav", "sepformer_track_0.wav", "sepformer_track_1.wav"):
        with wave.open(str(job / filename), "wb") as recording:
            recording.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            recording.writeframes(b"\x00\x00"*48000)
    (root / "web").mkdir()
    # Templates are test inputs, not fabricated inference output.
    import shutil
    shutil.copytree(ROOT / "web/recording-viewer", root / "web/recording-viewer")
    return PublicDemoConfig("data/source", "data/feedback", "demos/podcast-57min", (DemoExample("overlap_0001", job_id),))


def test_snapshot_is_allowlisted_independent_and_does_not_relabel_history() -> None:
    private = document()
    original = deepcopy(private)
    public = public_snapshot(private)
    assert private == original and public["read_only"] is True
    assert public["default_backend"] == "nemo_native"
    assert public["recovery"]["rejected"] == 1 and public["recovery"]["decisions"][0]["reason"] == "private_detail_not_exported"
    assert "PRIVATE_ABSOLUTE_PATH" not in json.dumps(public)
    assert "local_app_url" not in public and "configuration" not in public
    assert "not a new full-recording" in public["notice"]


def test_script_and_markdown_treat_transcript_content_as_data() -> None:
    public = public_snapshot(document())
    assert "</script>" not in script_json(public)
    assert json.loads(script_json(public)) == public
    markdown = transcript_markdown(public)
    assert "<script>fixture" not in markdown and "&lt;script&gt;fixture" in markdown


@pytest.mark.parametrize("changes", [{"duration": 0}, {"duration": float("nan")}, {"audio": "../private.mp3"}, {"variants": {}}])
def test_invalid_public_input_is_typed(changes: dict[str, object]) -> None:
    with pytest.raises(RecordingViewerError):
        public_snapshot({**document(), **changes})


def test_complete_export_contains_only_shareable_assets_and_preserves_source(tmp_path: Path) -> None:
    policy = fixture(tmp_path)
    source = (tmp_path / policy.recording_directory / "transcript.json").read_bytes()
    output = export_demo(tmp_path, policy)
    assert (tmp_path / policy.recording_directory / "transcript.json").read_bytes() == source
    for name in ("index.html", "examples.html", "audio.mp3", "TRANSCRIPT.md", "README.md", "results.json", "transcript.json"):
        assert (output / name).is_file()
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".md", ".json"}:
            assert "PRIVATE_ABSOLUTE_PATH" not in path.read_text(encoding="utf-8")
            assert b"\r\n" not in path.read_bytes()
    manifest = json.loads((output / "artifact-manifest.json").read_text())
    for name, identity in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == identity["sha256"]
    assert json.loads((output / "results.json").read_text())["full_recording"]["recovery_state"].startswith("historical_")
    assert "__RECORDING_DATA__" not in (output / "index.html").read_text(encoding="utf-8")
    with pytest.raises(RecordingViewerError, match="already exists"):
        export_demo(tmp_path, policy)


def test_wrong_audio_identity_does_not_publish_partial_output(tmp_path: Path) -> None:
    policy = fixture(tmp_path)
    (tmp_path / policy.recording_directory / "audio.mp3").write_bytes(b"different audio")
    with pytest.raises(RecordingViewerError, match="identity"):
        export_demo(tmp_path, policy)
    assert not (tmp_path / policy.output_directory).exists()


def test_unequal_clip_lengths_abort_atomically(tmp_path: Path) -> None:
    policy = fixture(tmp_path)
    track = tmp_path / "data/overlap-lab/runs" / ("1"*32) / "sepformer_track_1.wav"
    with wave.open(str(track), "wb") as recording:
        recording.setparams((1, 2, 16000, 0, "NONE", "not compressed"));recording.writeframes(b"\x00\x00"*16000)
    with pytest.raises(RecordingViewerError, match="identical sample"):
        export_demo(tmp_path, policy)
    assert not (tmp_path / policy.output_directory).exists()


@pytest.mark.parametrize("clip_id,job_id", [("../secret", "1"*32), ("overlap_0001", "../../x"), (True, "1"*32)])
def test_example_identifiers_are_not_paths(clip_id: str, job_id: str) -> None:
    with pytest.raises(RecordingViewerError):
        DemoExample(clip_id, job_id)
