"""Real HTTP and deterministic-provider tests for the private evaluation lab."""

from dataclasses import asdict, replace
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from overlap_lab.adapters.file_studies import FileStudyRepository
from overlap_lab.adapters.local_experiments import LocalExperimentProcessor
from overlap_lab.adapters.mock_adapter import MockExperimentProcessor
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabConflict, LabInputError
from overlap_lab.models import ClipSpec, ExperimentRequest, ReferenceAnnotation
from overlap_lab.services.lab_service import LabService
from overlap_lab.services.scoring import score_speakers, tokenize, word_errors
from scripts.serve_recording_viewer import RecordingViewerHandler


def config() -> LabConfig:
    return LabConfig.load(Path("configs/overlap_lab.json"))


def reference(reviewed: bool = True) -> ReferenceAnnotation:
    return ReferenceAnnotation(ClipSpec("test_clip", 9, 12, ("A", "B")), {"A": "Hello there", "B": "Good morning"}, reviewed)


def repository(tmp_path: Path) -> FileStudyRepository:
    folder = tmp_path / "recording"
    folder.mkdir()
    (folder / "processing.wav").write_bytes(b"fixture")
    (folder / "audio.wav").write_bytes(bytes(range(100)))
    (folder / "index.html").write_text("Fixture", encoding="utf-8")
    document = {"filename": "fixture.wav", "source_sha256": "a"*64, "duration": 20, "waveform": [.1,.2,.4],
                "audio": "audio.wav", "variants": {"nemo_native": {"turns": [
                    {"id": "a", "speaker": "A", "start": 0, "end": 12, "text": "Hello"},
                    {"id": "b", "speaker": "B", "start": 9, "end": 20, "text": "Hi"}]}}}
    (folder / "transcript.json").write_text(json.dumps(document), encoding="utf-8")
    return FileStudyRepository(folder, tmp_path / "uploads", tmp_path / "references", config())


def test_wer_alignment_is_explainable() -> None:
    score = word_errors("Hello my friend", "hello your friend today")
    assert (score.substitutions, score.deletions, score.insertions) == (1, 0, 1)
    assert score.wer == pytest.approx(2/3)
    assert [row["kind"] for row in score.alignment] == ["correct", "substitution", "correct", "insertion"]


def test_empty_reference_and_over_100_percent() -> None:
    assert word_errors("", "hallucinated words").wer is None
    assert word_errors("", "hallucinated words").insertions == 2
    assert word_errors("yes", "no never ever").wer == 3
    assert word_errors("one two", "").deletions == 2


def test_tokenization_and_draft_scoring() -> None:
    assert tokenize("Hello, DON’T! café 42.") == ["hello", "don't", "café", "42"]
    assert score_speakers(reference(False), {"A": "Hello", "B": "Good"}) is None
    assert score_speakers(reference(), {"A": "Hello there", "B": "Good morning"})["wer"] == 0


def test_studies_and_source_bound_annotation_save(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    study = repo.study("current")
    assert study["clips"][0]["start"] == 9
    assert study["clips"][0]["speakers"] == ["A", "B"]
    assert study["audio_url"] == "/audio.wav"
    assert "recording_folder" not in study
    saved = repo.save_reference("current", "a"*64, 0, reference())
    assert saved["revision"] == 1
    assert repo.annotations("current")["items"]["test_clip"]["reviewed"] is True
    with pytest.raises(LabConflict):
        repo.save_reference("current", "a"*64, 0, reference())
    with pytest.raises(LabConflict):
        repo.save_reference("current", "b"*64, 1, reference())


def test_replacing_current_recording_keeps_old_references_separate(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    repo.save_reference("current", "a"*64, 0, reference())
    path=repo.current / "transcript.json"
    data=json.loads(path.read_text())
    data["source_sha256"]="b"*64
    path.write_text(json.dumps(data))
    assert repo.annotations("current")["revision"] == 0
    repo.save_reference("current", "b"*64, 0, reference())
    assert len(list(repo.storage.glob("*.json"))) == 2


def test_completed_comparisons_survive_service_restart(tmp_path: Path) -> None:
    repo=repository(tmp_path)
    policy=config()
    job_id="1"*32
    folder=tmp_path / policy.storage_directory / "runs" / job_id
    folder.mkdir(parents=True)
    result={"recording_id":"current","source_hash":"a"*64,"variants":[],"request":asdict(ExperimentRequest("current",reference(),("sepformer--base",),1))}
    (folder / "result.json").write_text(json.dumps(result))
    processor=LocalExperimentProcessor(tmp_path,repo.current,repo.uploads,repo,policy)
    service=LabService(repo,processor,policy)
    try:
        assert service.get(job_id).status == "completed"
        assert service.history("current")[0].id == job_id
        assert processor.load("../../secret") is None
        assert processor.history("unknown") == []
    finally:
        service.close()


@pytest.mark.parametrize("recording", ["../secrets", "current/../../x", "C:/secret", "x"*32])
def test_recording_paths_are_not_accepted(tmp_path: Path, recording: str) -> None:
    with pytest.raises(LabInputError):
        repository(tmp_path).study(recording)


def test_validation_and_mock_run(tmp_path: Path) -> None:
    service = LabService(repository(tmp_path), MockExperimentProcessor(), config())
    try:
        with pytest.raises(LabInputError, match="unavailable"):
            service.submit(ExperimentRequest("current", reference(), ("mossformer--small",), 1))
        with pytest.raises(LabInputError, match="Context"):
            service.submit(ExperimentRequest("current", reference(), ("sepformer--base",), 100))
        job = service.submit(ExperimentRequest("current", reference(), ("sepformer--base",), 1))
    finally:
        service.close()
    assert service.get(job.id).status == "completed"
    assert service.get(job.id).result["mock"] is True
    assert service.history("current")[0].id == job.id


def test_wrong_bounds_speakers_and_huge_scoring_input(tmp_path: Path) -> None:
    service = LabService(repository(tmp_path), MockExperimentProcessor(), config())
    try:
        for invalid in [ReferenceAnnotation(ClipSpec("x", 9, 19, ("A", "B")), {"A":"", "B":""}, False),
                        ReferenceAnnotation(ClipSpec("x", 1, 3, ("A", "B")), {"A":"", "B":""}, False),
                        replace(reference(), texts={"A": "word "*501, "B": ""})]:
            with pytest.raises(LabInputError):
                service.validate_reference("current", invalid)
    finally:
        service.close()


def test_capability_discovery_never_claims_missing_models_ready(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    processor = LocalExperimentProcessor(Path.cwd(), repo.current, repo.uploads, repo, config())
    variants = processor.capabilities()
    assert len(variants) == 4
    assert all(not v["ready"] for v in variants if v["separator_id"] == "mossformer")


def test_real_http_lab_routes_and_security(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = LabService(repo, MockExperimentProcessor(), config())
    server = ThreadingHTTPServer(("127.0.0.1",0), partial(RecordingViewerHandler,
        directory=str(repo.current), lab=service, lab_config=config()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def post(path: str, body: object, origin: str | None = None) -> object:
        headers={"Content-Type":"application/json"}
        if origin:
            headers["Origin"]=origin
        with urlopen(Request(url+path,data=json.dumps(body).encode(),headers=headers)) as response:
            return json.load(response)

    try:
        with urlopen(url+"/lab/") as response:
            assert response.status == 200 and b"Make the overlap measurable" in response.read()
        with urlopen(url+"/lab/lab.js") as response:
            assert b"score_speakers" not in response.read()  # no server internals
        with urlopen(url+"/api/lab/bootstrap?recording=current") as response:
            bootstrap=json.load(response)
        assert bootstrap["annotations"]["revision"] == 0
        save={"recording_id":"current","source_hash":"a"*64,"revision":0,"reference":asdict(reference())}
        assert post("/api/lab/references",save)["revision"] == 1
        with pytest.raises(HTTPError) as failure:
            post("/api/lab/references",save)
        assert failure.value.code == 409
        with pytest.raises(HTTPError) as failure:
            post("/api/lab/references",save,"https://untrusted.example")
        assert failure.value.code == 403
        score=post("/api/lab/score",{"recording_id":"current","reference":asdict(reference()),"hypotheses":reference().texts})
        assert score["scores"]["wer"] == 0
        job=post("/api/lab/runs",asdict(ExperimentRequest("current",reference(),("sepformer--base",),1)))
        service.close()
        with urlopen(url+"/api/lab/jobs/"+job["id"]) as response:
            assert json.load(response)["status"] == "completed"
        for path in ("/lab/../../configs/overlap_lab.json", "/lab-audio/"+job["id"]+"/input.json"):
            with pytest.raises(HTTPError) as failure:
                urlopen(url+path)
            assert failure.value.code == 404
    finally:
        service.close()
        server.shutdown()
        server.server_close()
        thread.join()
