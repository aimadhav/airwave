"""Offline contract, persistence and HTTP tests for listening-only annotations."""

from array import array
from dataclasses import asdict, replace
from functools import partial
from http.server import ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest
from overlap_lab.adapters.file_studies import FileStudyRepository
from overlap_lab.adapters.listening_store import FileListeningReviewRepository
from overlap_lab.adapters.mock_adapter import MockExperimentProcessor, MockListeningReviewRepository
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabConflict, LabInputError
from overlap_lab.models import ClipSpec, ReferenceAnnotation
from overlap_lab.models.listening import ListeningReview
from overlap_lab.services.lab_service import LabService
from overlap_lab.services.listening_service import ListeningReviewService, starter_clips, voice_examples
from scripts.serve_recording_viewer import RecordingViewerHandler


def policy() -> LabConfig:
    return LabConfig.load(ROOT / "configs/overlap_lab.json")


def make_studies(tmp_path: Path) -> FileStudyRepository:
    """Generate an explicitly synthetic tone, not a fabricated inference result."""
    folder = tmp_path / "recording"
    folder.mkdir()
    with wave.open(str(folder / "audio.wav"), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(array("h", (round(6000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(24 * 16000))).tobytes())
    (folder / "processing.wav").write_bytes((folder / "audio.wav").read_bytes())
    (folder / "index.html").write_text("Synthetic listening QA fixture", encoding="utf-8")
    source_hash = hashlib.sha256((folder / "audio.wav").read_bytes()).hexdigest()
    data = {"filename": "[SYNTHETIC QA] single tone - not a real conversation.wav", "source_sha256": source_hash,
            "duration": 24, "audio": "audio.wav", "waveform": [.1, .2, .1],
            "recovery": {"decisions": [{"start": 3+i, "end": 4+i, "status": "unreviewed", "reason": "synthetic_fixture"} for i in range(12)]},
            "variants": {"nemo_native": {"turns": [{"id": "a", "speaker": "A", "start": 0, "end": 20},
                                                    {"id": "b", "speaker": "B", "start": 3, "end": 24}]}}}
    (folder / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    return FileStudyRepository(folder, tmp_path / "uploads", tmp_path / "references", policy())


def review(verdict: str = "one_speaker", heard: tuple[str, ...] = ()) -> ListeningReview:
    return ListeningReview(ClipSpec("overlap_0000", 3, 4, ("A", "B")), verdict, heard)


@pytest.mark.parametrize("verdict,heard", [("one_speaker", ()), ("one_speaker", ("A",)),
    ("two_overlapping", ("A", "B")), ("two_sequential", ("B", "A")), ("no_speech", ()), ("other_voices", ()), ("unsure", ())])
def test_listening_contract_round_trip(verdict: str, heard: tuple[str, ...]) -> None:
    value = review(verdict, heard)
    assert ListeningReview.from_mapping(json.loads(json.dumps(asdict(value)))) == value


@pytest.mark.parametrize("verdict,heard", [("invented", ()), ("one_speaker", ("A", "B")),
    ("one_speaker", ("C",)), ("two_overlapping", ("A",)), ("two_sequential", ()), ("no_speech", ("A",)),
    ("other_voices", ("B",)), ("unsure", ("A",)), ("one_speaker", ("A", "A"))])
def test_invalid_or_contradictory_labels(verdict: str, heard: tuple[str, ...]) -> None:
    with pytest.raises(LabInputError):
        review(verdict, heard)


@pytest.mark.parametrize("invalid", [[], None, {"verdict": "one_speaker", "heard_speakers": "A"}, {"verdict": 4}])
def test_invalid_json_review_is_typed(invalid: object) -> None:
    with pytest.raises(LabInputError):
        ListeningReview.from_mapping(invalid)


def test_store_is_source_bound_revision_safe_and_does_not_touch_references(tmp_path: Path) -> None:
    studies = make_studies(tmp_path)
    source = studies.study("current")["source_hash"]
    studies.save_reference("current", source, 0, ReferenceAnnotation(review().clip, {"A": "user words", "B": ""}, True))
    original = studies.annotations("current")
    store = FileListeningReviewRepository(studies, tmp_path / "listening", policy())
    saved = store.save("current", source, 0, review())
    assert saved["revision"] == 1 and saved["items"]["overlap_0000"]["heard_speakers"] == ()
    assert FileListeningReviewRepository(studies, tmp_path / "listening", policy()).reviews("current")["revision"] == 1
    assert studies.annotations("current") == original
    with pytest.raises(LabConflict):
        store.save("current", source, 0, review())
    with pytest.raises(LabConflict):
        store.save("current", "b" * 64, 1, review())
    data_path = studies.current / "transcript.json"
    data = json.loads(data_path.read_text())
    data["source_sha256"] = "b" * 64
    data_path.write_text(json.dumps(data))
    assert store.reviews("current")["revision"] == 0
    store.save("current", "b" * 64, 0, review("no_speech"))
    assert len(list(store.storage.glob("*.json"))) == 2


def test_service_accepts_false_overlap_and_mock_provider(tmp_path: Path) -> None:
    studies = make_studies(tmp_path)
    source = studies.study("current")["source_hash"]
    mock = MockListeningReviewRepository(source)
    service = ListeningReviewService(studies, mock, policy())
    result = service.save("current", source, 0, review())
    assert result["items"]["overlap_0000"]["verdict"] == "one_speaker"
    assert studies.annotations("current")["revision"] == 0
    data = service.bootstrap("current")
    assert len(data["starter_ids"]) == 12 and "existing_variants" not in data["study"]
    assert data["reviews"]["revision"] == 1
    for invalid in (replace(review(), clip=ClipSpec("not_detected", 3, 4, ("A", "B"))),
                    replace(review(), clip=ClipSpec("overlap_0000", 3, 5, ("A", "B")))):
        with pytest.raises(LabInputError):
            service.save("current", source, 1, invalid)
    with pytest.raises(LabInputError):
        service.save("current", source, True, review())


def test_sampling_spreads_unique_clips_and_handles_empty_or_short_sets() -> None:
    clips = [ClipSpec(f"c{i}", i, i+.8, ("A", "B")) for i in range(100)]
    chosen = starter_clips(clips, 100, 12, .6)
    assert len(chosen) == len(set(chosen)) == 12
    assert int(chosen[0][1:]) < 10 and int(chosen[-1][1:]) > 90
    assert starter_clips([], 10, 12, .6) == []
    assert starter_clips(clips[:1], 10, 12, 1) == ["c0"]


def test_voice_examples_exclude_overlaps_and_count_unique_speakers() -> None:
    regions = [{"speaker": "A", "start": 0, "end": 8}, {"speaker": "A", "start": 0, "end": 4},
               {"speaker": "B", "start": 5, "end": 12}]
    examples = voice_examples(regions, 3)
    assert examples["A"]["end"] <= 5 and examples["B"]["start"] >= 8
    assert voice_examples([], 3) == {}


def test_listening_http_and_security(tmp_path: Path) -> None:
    studies = make_studies(tmp_path)
    config = policy()
    lab = LabService(studies, MockExperimentProcessor(), config)
    listening = ListeningReviewService(studies, FileListeningReviewRepository(studies, tmp_path / "listening", config), config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(RecordingViewerHandler,
        directory=str(studies.current), lab=lab, lab_config=config, listening=listening))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def post(body: object, origin: str | None = None) -> object:
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        with urlopen(Request(url + "/api/lab/listening-reviews", data=json.dumps(body).encode(), headers=headers)) as response:
            return json.load(response)

    try:
        for asset in ("/lab/listen/", "/lab/listen/listen.css", "/lab/listen/listen.js"):
            with urlopen(url + asset) as response:
                assert response.status == 200
        with urlopen(url + "/api/lab/listening?recording=current") as response:
            bootstrap = json.load(response)
        body = {"recording_id": "current", "source_hash": bootstrap["study"]["source_hash"], "revision": 0, "review": asdict(review())}
        assert post(body)["revision"] == 1
        with pytest.raises(HTTPError) as failure:
            post(body)
        assert failure.value.code == 409
        with pytest.raises(HTTPError) as failure:
            post({**body, "revision": 1, "review": []})
        assert failure.value.code == 400
        with pytest.raises(HTTPError) as failure:
            post({**body, "revision": True})
        assert failure.value.code == 400
        with pytest.raises(HTTPError) as failure:
            post(body, "https://untrusted.example")
        assert failure.value.code == 403
        for invalid in ("/lab/listen/../../configs/overlap_lab.json", "/api/lab/listening?recording=../secret"):
            with pytest.raises(HTTPError):
                urlopen(url + invalid)
        assert studies.annotations("current")["revision"] == 0
    finally:
        server.shutdown()
        server.server_close()
        lab.close()
        thread.join()


if __name__ == "__main__":
    # Explicit synthetic, temporary browser-QA server. Never changes real studies.
    import argparse
    import tempfile
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="listening-review-qa-") as temporary:
        studies = make_studies(Path(temporary))
        config = policy()
        lab = LabService(studies, MockExperimentProcessor(), config)
        listening = ListeningReviewService(studies, MockListeningReviewRepository(studies.study("current")["source_hash"]), config)
        try:
            with ThreadingHTTPServer(("127.0.0.1", args.port), partial(RecordingViewerHandler,
                    directory=str(studies.current), lab=lab, lab_config=config, listening=listening)) as server:
                print(f"Synthetic browser QA: http://127.0.0.1:{server.server_port}/lab/listen/?recording=current", flush=True)
                server.serve_forever()
        finally:
            lab.close()
