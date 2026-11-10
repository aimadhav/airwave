from ai_layer.common.config_loader import load_config
from ai_layer.common.llm_provider import MockLLMProvider
from ai_layer.conversation_analysis.action_item_extractor import ActionItemExtractor
from ai_layer.conversation_analysis.citation_linker import CitationLinker
from ai_layer.conversation_analysis.decision_extractor import DecisionExtractor
from ai_layer.conversation_analysis.fact_extractor import FactExtractor
from ai_layer.conversation_analysis.service import ConversationAnalysisService
from ai_layer.conversation_analysis.statement_normalizer import StatementNormalizer
from ai_layer.common.stt_provider import MockSTTProvider
from ai_layer.speaker_verification.service import SpeakerVerificationService
from ai_layer.transcription.service import STTTranscriptionService
from tests.conftest import sample_clips, sample_session_metadata, sample_speakers


def _verified_transcript():
    config = load_config("configs/ai_layer_config.yaml")
    transcript = STTTranscriptionService(
        MockSTTProvider(), config.transcription
    ).process_session_clips("session_001", sample_clips())
    return SpeakerVerificationService(
        MockLLMProvider(), config.speaker_verification
    ).verify_speakers(transcript, sample_speakers())


def test_fact_extractor():
    verified = _verified_transcript()
    facts = FactExtractor(MockLLMProvider()).extract_facts(verified.turns)
    assert facts
    assert "contract" in facts[0]["statement"].lower()


def test_decision_extractor():
    verified = _verified_transcript()
    decisions = DecisionExtractor(MockLLMProvider()).extract_decisions(verified.turns)
    assert decisions
    assert "launch" in decisions[0]["decision"].lower()


def test_action_item_extractor():
    verified = _verified_transcript()
    tasks = ActionItemExtractor(MockLLMProvider()).extract_action_items(verified.turns)
    assert tasks
    assert tasks[0]["assignee_name"] == "David"


def test_statement_normalizer():
    normalized = StatementNormalizer(MockLLMProvider()).normalize_to_self_contained(
        "He will finish it tomorrow.",
        "David",
        "David will prepare the slide deck by Friday.",
    )
    assert normalized


def test_citation_linker():
    verified = _verified_transcript()
    turn = verified.turns[0]
    citation = CitationLinker.link_citation(turn)
    assert citation.turn_id == turn.turn_id
    assert citation.spoken_text == turn.text


def test_conversation_analysis_service():
    config = load_config("configs/ai_layer_config.yaml")
    verified = _verified_transcript()
    report = ConversationAnalysisService(
        MockLLMProvider(), config.conversation_analysis
    ).analyze_conversation(verified, sample_session_metadata())
    assert report.facts
    assert report.decisions
    assert report.action_items
