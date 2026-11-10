"""Tests verifying adherence to the Canonical Inter-Layer Data Contracts

Conforms to Section 3 of Decoupled_Implementation_Guide.md:
  - Contract 1: DSP Layer ➔ AI Layer (AudioClipPayload)
  - Contract 2: AI Layer ➔ Database Layer (KnowledgePayload)
  - Contract 3: Query & RAG Retrieval Interface (SearchRequest / SearchResponse)
"""

import json
from pathlib import Path
import pytest

from ai_layer.models.transcription_models import AudioClipPayload, AudioClipSegment
from ai_layer.models.knowledge_models import KnowledgePayload
from ai_layer.models.query_models import SearchRequest, SearchResponse
from ai_layer.pipeline_orchestrator import AILayerOrchestrator
from ai_layer.storage.sqlite_repository import SQLiteRepository
from ai_layer.common.embedding_provider import MockEmbeddingProvider
from tests.conftest import orchestrator, sample_speakers


@pytest.fixture
def temp_repo(tmp_path: Path) -> SQLiteRepository:
    db_file = tmp_path / "test_contracts.db"
    embed_provider = MockEmbeddingProvider(dimensions=1536)
    return SQLiteRepository(str(db_file), embedding_provider=embed_provider)


def test_contract_1_audio_clip_payload_conversion():
    payload_dict = {
        "session_id": "sess_101",
        "source_audio_path": "data/raw/meeting_2026_10_02.wav",
        "sample_rate": 16000,
        "channels": 1,
        "duration_seconds": 125.4,
        "segments": [
            {
                "segment_id": "seg_001",
                "clip_path": "data/processed/sessions/sess_101/clips/seg_001.wav",
                "start_time": 0.0,
                "end_time": 4.35,
                "preliminary_speaker_id": "SPK_VOICE_01",
                "acoustic_confidence": 0.88,
                "is_overlap_separated": False,
                "voice_embedding": [0.012, -0.045, 0.128],
            },
            {
                "segment_id": "seg_002",
                "clip_path": "data/processed/sessions/sess_101/clips/seg_002.wav",
                "start_time": 3.80,
                "end_time": 7.10,
                "preliminary_speaker_id": "SPK_VOICE_02",
                "acoustic_confidence": 0.74,
                "is_overlap_separated": True,
                "voice_embedding": [-0.034, 0.091, 0.052],
            },
        ],
    }

    # Verify deserialization from canonical Contract 1 JSON
    payload = AudioClipPayload.from_dict(payload_dict)
    assert payload.session_id == "sess_101"
    assert payload.sample_rate == 16000
    assert len(payload.segments) == 2
    assert payload.segments[0].segment_id == "seg_001"
    assert payload.segments[1].is_overlap_separated is True

    # Verify conversion to internal AudioClip objects
    clips = payload.to_audio_clips()
    assert len(clips) == 2
    assert clips[0].clip_id == "seg_001"
    assert clips[0].time_range.start_time_seconds == 0.0
    assert clips[0].time_range.end_time_seconds == 4.35

    # Verify serialization back to dict
    serialized = payload.to_dict()
    assert serialized["session_id"] == "sess_101"
    assert len(serialized["segments"]) == 2


def test_contract_1_orchestrator_processing():
    orch = orchestrator()
    speakers = sample_speakers()
    payload = AudioClipPayload(
        session_id="sess_dsp_test",
        source_audio_path="data/sample_meeting.wav",
        sample_rate=16000,
        channels=1,
        duration_seconds=30.0,
        segments=[
            AudioClipSegment(
                segment_id="seg_001",
                clip_path="clip_001.wav",
                start_time=0.0,
                end_time=3.5,
                preliminary_speaker_id="Speaker_1",
            ),
            AudioClipSegment(
                segment_id="seg_002",
                clip_path="clip_002.wav",
                start_time=4.0,
                end_time=9.0,
                preliminary_speaker_id="Speaker_2",
            ),
        ],
    )

    result = orch.process_dsp_payload(
        payload=payload,
        session_title="DSP Integration Test Meeting",
        known_speakers=speakers,
        workspace_id="ws_test",
    )

    assert result.session_summary is not None
    assert result.verified_transcript is not None
    assert len(result.verified_transcript.turns) > 0


def test_contract_2_and_3_storage_and_rag(temp_repo: SQLiteRepository):
    orch = orchestrator()
    speakers = sample_speakers()
    orch._repository = temp_repo

    payload = AudioClipPayload(
        session_id="sess_contract_2",
        source_audio_path="data/raw/meeting_2026_10_02.wav",
        sample_rate=16000,
        channels=1,
        duration_seconds=60.0,
        segments=[
            AudioClipSegment(
                segment_id="seg_001",
                clip_path="clip_001.wav",
                start_time=0.0,
                end_time=4.35,
                preliminary_speaker_id="Speaker_1",
            ),
        ],
    )

    orch.process_dsp_payload(
        payload=payload,
        session_title="Contract 2 & 3 Validation Session",
        known_speakers=speakers,
        workspace_id="ws_default",
        project_id="proj_launch",
    )

    # --- Contract 2: Export KnowledgePayload ---
    knowledge_payload = orch.export_knowledge_payload("sess_contract_2", workspace_id="ws_default")
    assert isinstance(knowledge_payload, KnowledgePayload)
    k_dict = knowledge_payload.to_dict()

    # Verify canonical Contract 2 structure
    assert "session" in k_dict
    assert k_dict["session"]["session_id"] == "sess_contract_2"
    assert k_dict["session"]["workspace_id"] == "ws_default"
    assert "utterances" in k_dict
    assert len(k_dict["utterances"]) > 0
    assert "content_units" in k_dict
    assert "edges" in k_dict
    assert "change_logs" in k_dict

    # --- Contract 3: Hybrid Search Retrieval ---
    search_req = SearchRequest(
        workspace_id="ws_default",
        project_id="proj_launch",
        query_text="launch schedule",
        top_k=3,
        include_superseded=False,
        walk_edges=True,
    )
    search_res = orch.search_contract_rag(search_req)

    assert isinstance(search_res, SearchResponse)
    res_dict = search_res.to_dict()
    assert "results" in res_dict
    if res_dict["results"]:
        first = res_dict["results"][0]
        assert "content_id" in first
        assert "similarity_score" in first
        assert "recency_score" in first
        assert "combined_rank_score" in first
        assert "source_utterances" in first
        assert "connected_edges" in first
