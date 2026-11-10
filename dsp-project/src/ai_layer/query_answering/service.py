from ai_layer.common.config_loader import QueryAnsweringConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.analysis_models import AnalysisReport
from ai_layer.models.query_models import QueryAnswer, UserQuery
from ai_layer.models.speaker_models import VerifiedTranscript
from ai_layer.query_answering.answer_synthesizer import AnswerSynthesizer
from ai_layer.query_answering.citation_resolver import CitationResolver
from ai_layer.query_answering.in_memory_retriever import InMemoryRetriever
from ai_layer.query_answering.query_intent_analyzer import QueryIntentAnalyzer


class UserQueryService:
    def __init__(self, llm_provider: LLMProvider, config: QueryAnsweringConfig) -> None:
        self._top_k = config.retrieval_top_k
        self._intent_analyzer = QueryIntentAnalyzer(llm_provider)
        self._retriever = InMemoryRetriever(config)
        self._citation_resolver = CitationResolver()
        self._answer_synthesizer = AnswerSynthesizer(llm_provider, config)

    def answer_query(
        self,
        query: UserQuery,
        transcripts: list[VerifiedTranscript],
        reports: list[AnalysisReport],
    ) -> QueryAnswer:
        if query.target_session_id:
            transcripts = [
                transcript
                for transcript in transcripts
                if transcript.session_id == query.target_session_id
            ]
            reports = [
                report for report in reports if report.session_id == query.target_session_id
            ]

        intent = self._intent_analyzer.analyze_intent(query)
        turn_snippets = self._retriever.retrieve_relevant_turns(
            intent, transcripts, query_text=query.query_text
        )
        insight_snippets = self._retriever.retrieve_relevant_insights(
            intent, reports, query_text=query.query_text
        )
        
        # Combine all candidates and sort strictly by relevance score descending
        all_snippets = turn_snippets + insight_snippets
        all_snippets.sort(key=lambda item: item.relevance_score, reverse=True)

        # Deduplicate identical or near-identical snippets
        seen_texts: set[str] = set()
        deduped: list[RetrievedSnippet] = []
        for s in all_snippets:
            norm_content = " ".join(s.content.lower().split())
            if norm_content not in seen_texts:
                seen_texts.add(norm_content)
                deduped.append(s)

        combined = deduped[: self._top_k]
        citations = self._citation_resolver.resolve_citations(combined)
        return self._answer_synthesizer.synthesize_answer(query, combined, citations)
