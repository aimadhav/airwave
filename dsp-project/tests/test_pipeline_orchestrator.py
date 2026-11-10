from ai_layer.models.query_models import UserQuery
from tests.conftest import orchestrator, sample_clips, sample_session_metadata, sample_speakers


def test_pipeline_orchestrator_end_to_end():
    result = orchestrator().process_audio_session(
        sample_session_metadata(),
        sample_clips(),
        sample_speakers(),
    )

    assert result.verified_transcript.turns
    assert result.analysis_report.facts
    assert result.knowledge_store.records
    assert result.session_summary.overview
    assert result.project_summary is not None

    answer = orchestrator().query_session_memory(
        "What did we decide about the launch date?",
        [result.verified_transcript],
        [result.analysis_report],
    )
    assert answer.is_found
    assert answer.citations
