from pathlib import Path

from ai_layer.common.config_loader import load_config
from ai_layer.common.embedding_provider import MockEmbeddingProvider
from ai_layer.models.speaker_models import SpeakerProfile
from ai_layer.pipeline_orchestrator import AILayerOrchestrator
from ai_layer.storage.sqlite_repository import SQLiteRepository
from tests.conftest import sample_clips, sample_session_metadata, sample_speakers


def test_sqlite_repository_initialization_and_crud(tmp_path: Path):
    db_file = str(tmp_path / "test_layer.db")
    embedder = MockEmbeddingProvider(dimensions=1536)
    repo = SQLiteRepository(db_path=db_file, embedding_provider=embedder)

    # Verify speaker upsert
    speakers = [
        SpeakerProfile(speaker_id="spk_1", canonical_name="Alice", base_confidence=0.8),
        SpeakerProfile(speaker_id="spk_2", canonical_name="Bob", base_confidence=0.75),
    ]
    repo.upsert_speakers(speakers)

    # Process session using orchestrator
    config = load_config("configs/ai_layer_config.yaml")
    orchestrator = AILayerOrchestrator(config=config, repository=repo)

    result = orchestrator.process_audio_session(
        sample_session_metadata(),
        sample_clips(),
        sample_speakers(),
    )

    # Save session
    workspace_id = "test_workspace"
    orchestrator.save_session_result(
        workspace_id=workspace_id,
        session_metadata=sample_session_metadata(),
        result=result,
        audio_path="test_audio.wav",
    )

    # Verify session listed
    sessions = repo.list_session_ids(workspace_id)
    assert "session_001" in sessions

    # Verify transcript loaded
    transcript = repo.load_verified_transcript("session_001")
    assert transcript is not None
    assert transcript.session_id == "session_001"
    assert len(transcript.turns) > 0

    # Verify analysis report loaded
    report = repo.load_analysis_report("session_001")
    assert report.session_id == "session_001"
    assert len(report.facts) > 0
    assert len(report.decisions) > 0
    assert len(report.action_items) > 0

    # Verify knowledge store loaded
    store = repo.load_knowledge_store(workspace_id)
    assert len(store.records) > 0

    # Verify vector search
    units = repo.search_content_units(
        workspace_id=workspace_id,
        query_text="launch date",
        top_k=3,
    )
    assert len(units) > 0

    # Verify query_workspace on orchestrator
    answer = orchestrator.query_workspace(
        query_text="What did we decide about the launch date?",
        workspace_id=workspace_id,
    )
    assert answer.is_found
    assert len(answer.citations) > 0
