from ai_layer.common.config_loader import load_config
from ai_layer.common.llm_provider import MockLLMProvider
from ai_layer.models.query_models import UserQuery
from ai_layer.query_answering.answer_synthesizer import AnswerSynthesizer
from ai_layer.query_answering.citation_resolver import CitationResolver
from ai_layer.query_answering.in_memory_retriever import InMemoryRetriever
from ai_layer.query_answering.query_intent_analyzer import QueryIntentAnalyzer
from ai_layer.query_answering.service import UserQueryService
from tests.conftest import orchestrator, sample_session_metadata, sample_clips, sample_speakers


def _processed_session():
    result = orchestrator().process_audio_session(
        sample_session_metadata(),
        sample_clips(),
        sample_speakers(),
    )
    return result


def test_query_intent_analyzer():
    intent = QueryIntentAnalyzer(MockLLMProvider()).analyze_intent(
        UserQuery("What did we decide about the launch date?")
    )
    assert intent.intent_category == "decision"
    assert intent.target_topic == "launch date"


def test_in_memory_retriever():
    result = _processed_session()
    config = load_config("configs/ai_layer_config.yaml")
    retriever = InMemoryRetriever(config.query_answering)
    intent = QueryIntentAnalyzer(MockLLMProvider()).analyze_intent(
        UserQuery("What tasks did Sarah commit to?")
    )
    snippets = retriever.retrieve_relevant_insights(intent, [result.analysis_report])
    assert snippets


def test_citation_resolver():
    result = _processed_session()
    config = load_config("configs/ai_layer_config.yaml")
    retriever = InMemoryRetriever(config.query_answering)
    intent = QueryIntentAnalyzer(MockLLMProvider()).analyze_intent(
        UserQuery("What did we decide about the launch date?")
    )
    snippets = retriever.retrieve_relevant_insights(intent, [result.analysis_report])
    citations = CitationResolver.resolve_citations(snippets)
    assert citations
    assert ":" in citations[0].timestamp_str


def test_answer_synthesizer():
    config = load_config("configs/ai_layer_config.yaml")
    result = _processed_session()
    retriever = InMemoryRetriever(config.query_answering)
    intent = QueryIntentAnalyzer(MockLLMProvider()).analyze_intent(
        UserQuery("What did we decide about the launch date?")
    )
    snippets = retriever.retrieve_relevant_insights(intent, [result.analysis_report])
    citations = CitationResolver.resolve_citations(snippets)
    answer = AnswerSynthesizer(MockLLMProvider(), config.query_answering).synthesize_answer(
        UserQuery("What did we decide about the launch date?"),
        snippets,
        citations,
    )
    assert answer.is_found
    assert answer.answer_text


def test_answer_synthesizer_fallback():
    config = load_config("configs/ai_layer_config.yaml")
    answer = AnswerSynthesizer(MockLLMProvider(), config.query_answering).synthesize_answer(
        UserQuery("What is the office address?"),
        [],
        [],
    )
    assert not answer.is_found
    assert answer.answer_text == config.query_answering.fallback_message


def test_user_query_service():
    result = _processed_session()
    config = load_config("configs/ai_layer_config.yaml")
    answer = UserQueryService(MockLLMProvider(), config.query_answering).answer_query(
        UserQuery("What did we decide about the launch date?"),
        [result.verified_transcript],
        [result.analysis_report],
    )
    assert answer.is_found


def test_user_query_service_multi_speaker():
    from ai_layer.models.base_models import TimeRange
    from ai_layer.models.speaker_models import VerifiedTranscript, VerifiedTurn
    from ai_layer.models.analysis_models import ActionItem, AnalysisReport, UtteranceCitation

    config = load_config("configs/ai_layer_config.yaml")
    vt = VerifiedTranscript(
        session_id="multi_test_sess",
        turns=[
            VerifiedTurn(
                turn_id="t1",
                speaker_id="spk_rahul",
                speaker_name="Rahul",
                text="We can use RANSAC to filter out noisy outliers in spatial mapping.",
                time_range=TimeRange(0.0, 5.0),
            ),
            VerifiedTurn(
                turn_id="t2",
                speaker_id="spk_aman",
                speaker_name="Aman",
                text="Hough transform works better for lines.",
                time_range=TimeRange(5.5, 10.0),
            ),
        ],
    )
    report = AnalysisReport(
        session_id="multi_test_sess",
        action_items=[
            ActionItem(
                task_id="t_deck",
                task_description="Prepare the slide deck",
                assignee_name="David",
                citation=UtteranceCitation(
                    turn_id="t3",
                    speaker_name="David",
                    time_range=TimeRange(15.0, 20.0),
                    spoken_text="David will prepare the slide deck",
                ),
            )
        ],
    )

    service = UserQueryService(MockLLMProvider(), config.query_answering)
    ans = service.answer_query(
        UserQuery("What did Rahul say about RANSAC?"),
        [vt],
        [report],
    )

    assert ans.is_found
    assert len(ans.citations) > 0
    assert ans.citations[0].speaker_name == "Rahul"
    assert "RANSAC" in ans.citations[0].verbatim_quote
    # Confirm David's task was rejected
    assert all("David" not in c.speaker_name for c in ans.citations)


def test_user_query_service_unrelated_query():
    result = _processed_session()
    config = load_config("configs/ai_layer_config.yaml")
    answer = UserQueryService(MockLLMProvider(), config.query_answering).answer_query(
        UserQuery("What is the capital of Australia?"),
        [result.verified_transcript],
        [result.analysis_report],
    )
    assert not answer.is_found
    assert len(answer.citations) == 0
    assert answer.answer_text == config.query_answering.fallback_message


def test_citation_provenance_and_session_id():
    from ai_layer.models.base_models import TimeRange
    from ai_layer.models.query_models import RetrievedSnippet
    from ai_layer.query_answering.citation_resolver import CitationResolver

    snippet = RetrievedSnippet(
        snippet_id="t1",
        content="We finalized the migration.",
        speaker_name="Alice",
        time_range=TimeRange(125.0, 130.0),
        session_id="sess_101",
        session_title="Architecture Review",
    )
    citations = CitationResolver.resolve_citations([snippet])
    assert len(citations) == 1
    assert citations[0].speaker_name == "Alice"
    assert citations[0].session_id == "sess_101"
    assert citations[0].session_title == "Architecture Review"
    assert citations[0].timestamp_str == "02:05"

