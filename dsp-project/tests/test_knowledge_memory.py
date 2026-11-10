from ai_layer.common.config_loader import load_config
from ai_layer.common.llm_provider import MockLLMProvider
from ai_layer.knowledge_memory.service import KnowledgeMemoryService
from tests.conftest import orchestrator, sample_clips, sample_session_metadata, sample_speakers


def test_knowledge_memory_integrates_insights():
    result = orchestrator().process_audio_session(
        sample_session_metadata(),
        sample_clips(),
        sample_speakers(),
    )
    config = load_config("configs/ai_layer_config.yaml")
    store = KnowledgeMemoryService(MockLLMProvider(), config.knowledge_memory).integrate_session_insights(
        result.analysis_report
    )
    assert store.records
    assert store.change_log
